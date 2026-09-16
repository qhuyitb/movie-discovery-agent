"""The assistant itself.

Two interchangeable backends over one shared tool layer:

* `HeuristicAgent` - the default. A rule-based planner (`planner.py`) picks an
  intent, runs the relevant tools, and `nlg.py` renders the evidence. Fully
  offline, deterministic, and every claim is traceable to a tool result.
* `LLMAgent` - the same tools handed to a real model as function schemas, with a
  system prompt that forbids answering from the model's own movie knowledge.
  Switched on with `--backend openai|anthropic` plus an API key.

Both return an `AgentTurn` carrying the answer *and* the tool trace, so the
reasoning is inspectable rather than asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import nlg
from .planner import Understanding, understand
from .tools import TOOL_SCHEMAS, MovieTools, ToolCall

SYSTEM_PROMPT = """\
You are a movie discovery assistant for a MovieLens-derived catalogue of 5,135 films \
rated by 610 users.

Hard rules:
1. Every factual claim about a film, a user or a rating MUST come from a tool result in \
this conversation. You have no other knowledge of these films. If a tool has not told you \
something, say you don't know or look it up.
2. Personalise. Once you know the user's id, consult their profile and their neighbours \
before recommending. Never give a generic "popular films" list to an identified user \
without saying that is what you are doing.
3. Multi-step questions need multi-step investigation. "What do people like me think of X" \
means: find similar users, then look up their ratings of X, then synthesise.
4. Quote the evidence: rating counts, neighbour averages, the user's own past ratings. \
Prefer "18 of your closest neighbours rated it 4.4" over "critics loved it".
5. Be honest about thin evidence. A 5.0 from 3 ratings is not a strong signal, and if none \
of the user's neighbours has seen a film, say so.
6. Keep answers compact: a short lead-in, then the films with their evidence.
"""


@dataclass
class ConversationState:
    user_id: int | None = None
    last_movie: str | None = None
    last_recommendations: list[dict] = field(default_factory=list)
    turns: int = 0


@dataclass
class AgentTurn:
    text: str
    trace: list[ToolCall]
    intent: str = ""
    notes: list[str] = field(default_factory=list)

    def trace_lines(self) -> list[str]:
        return [f"{c.summary()}  [{c.ms:.0f} ms]" for c in self.trace]


class BaseAgent:
    def __init__(self, tools: MovieTools, user_id: int | None = None):
        self.tools = tools
        self.state = ConversationState(user_id=user_id)

    def _remember(self, recs: list[dict]) -> None:
        if recs:
            self.state.last_recommendations = recs
            self.state.last_movie = recs[0]["title"]


# ---------------------------------------------------------------------------
# Offline backend
# ---------------------------------------------------------------------------
class HeuristicAgent(BaseAgent):
    """Rule-based planner + grounded templates. No network, no API key."""

    def chat(self, message: str) -> AgentTurn:
        self.tools.reset_trace()
        self.state.turns += 1
        u = understand(message, self.tools.data, self.state)
        if u.user_id is not None:
            if not self.tools.data.has_user(u.user_id):
                # Do not adopt an id that isn't in the data - the rest of the
                # conversation would then fail one tool call at a time.
                return AgentTurn(
                    text=(f"There's no user {u.user_id} in this dataset - ids run from 1 to 610. "
                          + (f"You're still user {self.state.user_id}." if self.state.user_id
                             else "Give me one in that range and I'll read their history.")),
                    trace=[], intent="identify", notes=u.notes)
            self.state.user_id = u.user_id

        handler = getattr(self, f"_do_{u.intent}", None)
        if handler is None:
            text = self._help()
        elif u.intent not in ("movie_info", "similar_movies", "content_search",
                              "unresolved_movie", "unknown") and self.state.user_id is None:
            text = ("Tell me your user id first (for example \"I'm user 30\") - without it I'd just be "
                    "reciting popular films rather than reading your history.")
        else:
            text = handler(u)
        return AgentTurn(text=text, trace=list(self.tools.trace), intent=u.intent, notes=u.notes)

    # -- handlers ----------------------------------------------------------
    def _do_identify(self, u: Understanding) -> str:
        p = self.tools.call("get_user_profile", {"user_id": self.state.user_id})
        if "error" in p:
            return p["error"]
        return nlg.profile(p) + "\n\nAsk me what to watch, or about a specific film."

    _do_profile = _do_identify

    def _do_recommend(self, u: Understanding) -> str:
        payload = self.tools.call("recommend_for_user", {
            "user_id": self.state.user_id, "n": 5,
            "include_genres": u.include_genres or None,
            "exclude_genres": u.exclude_genres or None,
            "similar_to": u.liked_seed or None,
            "year_from": u.year_from, "year_to": u.year_to,
        })
        if "error" in payload:
            return payload["error"]
        recs = payload.get("recommendations", [])
        # Second hop: pull the evidence behind the top picks so the answer can justify itself.
        reasons = {}
        for rec in recs[:3]:
            why = self.tools.call("explain_recommendation",
                                  {"user_id": self.state.user_id, "movie": rec["title"]})
            if "error" not in why:
                reasons[rec["movieId"]] = why
        prof = self.tools.call("get_user_profile", {"user_id": self.state.user_id})
        self._remember(recs)
        return nlg.recommendations(payload, reasons, prof if "error" not in prof else None)

    def _do_content_search(self, u: Understanding) -> str:
        payload = self.tools.call("search_movies", {
            "query": u.query or "", "n": 6,
            "include_genres": u.include_genres or None,
            "exclude_genres": u.exclude_genres or None,
            "year_from": u.year_from, "year_to": u.year_to,
            "min_ratings": 10,
            "for_user": self.state.user_id,
        })
        res = payload.get("results", [])
        self._remember(res)
        text = nlg.search_results(payload)
        if self.state.user_id is None:
            text += "\n\n(Tell me your user id and I'll rank these against your history instead of the crowd's.)"
        return text

    def _do_peer_opinion(self, u: Understanding) -> str:
        movie = u.movies[0]
        p = self.tools.call("how_similar_users_rated",
                            {"user_id": self.state.user_id, "movie": movie, "n_users": 30})
        if "error" not in p:
            self.state.last_movie = p["movie"]["title"]
        return nlg.peer_opinion(p)

    def _do_explain(self, u: Understanding) -> str:
        movie = u.movies[0] if u.movies else (
            self.state.last_recommendations[0]["title"] if self.state.last_recommendations else None)
        if movie is None:
            return "Which film do you want me to justify? I don't have one from earlier in this conversation."
        why = self.tools.call("explain_recommendation", {"user_id": self.state.user_id, "movie": movie})
        if "error" not in why:
            self.state.last_movie = why["movie"]["title"]
        return nlg.explanation(why)

    def _do_blind_spots(self, u: Understanding) -> str:
        return nlg.blind_spots(self.tools.call("find_blind_spots", {"user_id": self.state.user_id}))

    def _do_similar_users(self, u: Understanding) -> str:
        return nlg.similar_users(self.tools.call("find_similar_users",
                                                 {"user_id": self.state.user_id, "n": 5}))

    def _do_similar_movies(self, u: Understanding) -> str:
        s = self.tools.call("find_similar_movies", {"movie": u.movies[0], "n": 6, "basis": "both"})
        if "error" not in s:
            self.state.last_movie = s["movie"]
        return nlg.similar_movies(s)

    def _do_movie_info(self, u: Understanding) -> str:
        m = self.tools.call("get_movie_details", {"movie": u.movies[0]})
        peers = None
        if "error" not in m:
            self.state.last_movie = m["title"]
            if self.state.user_id is not None:
                peers = self.tools.call("how_similar_users_rated",
                                        {"user_id": self.state.user_id, "movie": m["title"]})
        return nlg.movie_details(m, peers)

    def _do_unresolved_movie(self, u: Understanding) -> str:
        """The user named a film we can't find. Say so; don't answer a different question."""
        title = u.unresolved_title or ""
        near = self.tools.data.search_titles(title, 4)
        lines = [f"I can't find a film called {title!r} in this catalogue. This is a filtered "
                 f"MovieLens subset of 5,135 films; some well-known titles were dropped "
                 f"upstream because they had no plot text."]
        if near:
            lines.append("Closest titles I do have: "
                         + ", ".join(self.tools.data.title_of(m) for m in near) + ".")
        else:
            lines.append("Nothing in the catalogue has a similar title either.")
        lines.append("Name one of those and I'll check what users with your taste made of it.")
        return "\n".join(lines)

    def _do_unknown(self, u: Understanding) -> str:
        return ("I didn't get a clear question out of that, and I'd rather say so than guess.\n\n"
                + self._help())

    def _help(self) -> str:
        return (
            "I look things up in the ratings and plot data rather than guessing. Try:\n"
            "  - \"I'm user 30\"\n"
            "  - \"What should I watch tonight?\"\n"
            "  - \"I want a dark psychological thriller with a twist\"\n"
            "  - \"What do people with similar taste to mine think about Pulp Fiction?\"\n"
            "  - \"Why do you think I'd like that?\"\n"
            "  - \"What's my blind spot?\""
        )


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
class LLMAgent(BaseAgent):
    """Same tools, planned by a real model. Needs an API key; see README_SOLUTION.md."""

    MAX_STEPS = 6

    def __init__(self, tools: MovieTools, client, user_id: int | None = None, backend: str = "anthropic"):
        super().__init__(tools, user_id)
        self.client = client
        self.backend = backend
        self.messages: list[dict] = []

    def chat(self, message: str) -> AgentTurn:
        self.tools.reset_trace()
        self.state.turns += 1
        if self.state.user_id is not None and self.state.turns == 1:
            message = f"(I am user {self.state.user_id}.) {message}"
        self.messages.append({"role": "user", "content": message})

        final = ""
        for _ in range(self.MAX_STEPS):
            reply = self.client.complete(SYSTEM_PROMPT, self.messages, TOOL_SCHEMAS)
            if not reply.tool_calls:
                final = reply.text
                self.messages.append({"role": "assistant", "content": reply.text})
                break
            self._append_assistant(reply)
            for call in reply.tool_calls:
                result = self.tools.call(call["name"], call["arguments"])
                self._append_tool_result(call, result)
        else:
            final = "I ran out of investigation steps. Try asking something narrower."
        return AgentTurn(text=final, trace=list(self.tools.trace), intent="llm")

    # The two providers disagree about message shape; keep the difference here.
    def _append_assistant(self, reply) -> None:
        if self.backend == "anthropic":
            self.messages.append({"role": "assistant", "content": reply.raw.content})
        else:
            self.messages.append(reply.raw.model_dump(exclude_none=True))

    def _append_tool_result(self, call: dict, result) -> None:
        payload = json.dumps(result, default=str)[:12000]
        if self.backend == "anthropic":
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call["id"], "content": payload}]})
        else:
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": payload})


def build_agent(backend: str = "heuristic", user_id: int | None = None,
                model: str | None = None, base_url: str | None = None,
                tools: MovieTools | None = None) -> BaseAgent:
    tools = tools or MovieTools.bootstrap()
    if backend == "heuristic":
        return HeuristicAgent(tools, user_id=user_id)
    from .llm import build_client
    return LLMAgent(tools, build_client(backend, model, base_url), user_id=user_id, backend=backend)
