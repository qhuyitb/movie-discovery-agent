"""Language understanding for the offline (no-LLM) backend.

This is deliberately a small, legible rule system rather than a pretend LLM. It
does three jobs:

1. classify the request into one of the intents the tool layer can serve,
2. pull out slots - user id, film titles, genre constraints, year windows,
3. resolve pronouns ("why would I like *that*?") against conversation state.

It is the weakest component in the system and the report says so plainly. The
LLM backend replaces exactly this file and nothing else - the tools, the
evidence and the grounding rules are shared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data import MovieData, normalise_title

# --------------------------------------------------------------------- genres
GENRE_SYNONYMS: dict[str, list[str]] = {
    "Action": ["action", "action movie", "action films"],
    "Adventure": ["adventure"],
    "Animation": ["animated", "animation", "cartoon", "cartoons", "anime"],
    "Children": ["kids", "children", "family friendly", "family movie"],
    "Comedy": ["comedy", "comedies", "funny", "comedic", "laugh"],
    "Crime": ["crime", "gangster", "mob movie", "heist"],
    "Documentary": ["documentary", "documentaries", "docs"],
    "Drama": ["drama", "dramas", "dramatic"],
    "Fantasy": ["fantasy"],
    "Film-Noir": ["noir", "film noir", "film-noir"],
    "Horror": ["horror", "scary", "slasher", "frightening"],
    "Musical": ["musical", "musicals"],
    "Mystery": ["mystery", "mysteries", "whodunit"],
    "Romance": ["romance", "romantic", "rom com", "romcom", "love story"],
    "Sci-Fi": ["sci fi", "sci-fi", "scifi", "science fiction", "space opera"],
    "Thriller": ["thriller", "thrillers", "suspense", "suspenseful"],
    "War": ["war movie", "war film", "wartime"],
    "Western": ["western", "westerns"],
}

NEGATION_CUES = [
    "tired of", "sick of", "bored of", "no more", "not in the mood for", "don't want",
    "dont want", "do not want", "without", "avoid", "except", "but not", "no ",
    "not ", "skip", "hate", "can't stand", "cant stand", "less ", "nothing ",
]

STOPWORD_PREFIXES = [
    "i want", "i'd like", "i would like", "i am looking for", "i'm looking for",
    "looking for", "find me", "show me", "give me", "recommend me", "recommend",
    "suggest", "can you find", "i feel like", "i'm in the mood for", "im in the mood for",
    "something", "anything",
]

PRONOUNS = {"that", "it", "this", "those", "them", "that one"}


@dataclass
class Understanding:
    intent: str
    user_id: int | None = None
    movies: list[str] = field(default_factory=list)
    include_genres: list[str] = field(default_factory=list)
    exclude_genres: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    query: str = ""
    liked_seed: list[str] = field(default_factory=list)
    used_pronoun: bool = False
    unresolved_title: str | None = None
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------- extraction
def _find_genres(text: str) -> tuple[list[str], list[str]]:
    """Locate genre mentions and decide include vs exclude from nearby negation cues."""
    include, exclude = [], []
    for genre, words in GENRE_SYNONYMS.items():
        for w in words:
            for m in re.finditer(rf"\b{re.escape(w)}\b", text):
                window = text[max(0, m.start() - 30): m.start()]
                negated = any(cue in window for cue in NEGATION_CUES)
                (exclude if negated else include).append(genre)
                break
    return sorted(set(include) - set(exclude)), sorted(set(exclude))


def _find_titles(text: str, data: MovieData, max_words: int = 7) -> list[str]:
    """Longest-match scan of the message against known titles.

    Single-word titles must be capitalised in the original text, otherwise films
    called "Up" or "Heat" fire on ordinary sentences.
    """
    raw_tokens = text.split()
    tokens = [re.sub(r"[^A-Za-z0-9']+", " ", t).strip() for t in raw_tokens]
    found, i = [], 0
    while i < len(tokens):
        hit = None
        for span in range(min(max_words, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i: i + span]).strip()
            if len(phrase) < 2:
                continue
            key = normalise_title(phrase)
            if not key or key in {"the", "a", "an"}:
                continue
            mid = data._norm_titles.get(key)
            if mid is None:
                continue
            if span == 1 and not (raw_tokens[i][:1].isupper() and len(phrase) >= 4):
                continue
            hit = (phrase, span)
            break
        if hit:
            found.append(hit[0])
            i += hit[1]
        else:
            i += 1
    return found


def _find_years(text: str) -> tuple[int | None, int | None]:
    if m := re.search(r"\b(19\d{2}|20\d{2})s?\b.{0,12}\b(?:or|to|-|through)\b.{0,6}(19\d{2}|20\d{2})", text):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"\b(?:after|since|newer than|post)\s+(19\d{2}|20\d{2})", text):
        return int(m.group(1)), None
    if m := re.search(r"\b(?:before|older than|pre[- ])\s*(19\d{2}|20\d{2})", text):
        return None, int(m.group(1))
    if m := re.search(r"\b(19\d0|20\d0)s\b", text):
        decade = int(m.group(1))
        return decade, decade + 9
    return None, None


def _clean_query(text: str) -> str:
    q = text
    for _ in range(3):                       # "I want something dark" -> "dark"
        for p in STOPWORD_PREFIXES:
            q = re.sub(rf"^\s*{re.escape(p)}\b", " ", q, flags=re.I).strip()
    q = re.sub(r"^\s*(?:a|an|the|some)\b", " ", q, flags=re.I).strip()
    q = re.sub(r"\b(?:i'?m|i am)\s+user\s+\d+\b", " ", q, flags=re.I)
    q = re.sub(r"\buser\s+\d+\b", " ", q, flags=re.I)
    q = re.sub(r"[?!.]+$", "", q)
    return " ".join(q.split())


# ----------------------------------------------------------------- classifier
def understand(message: str, data: MovieData, state) -> Understanding:
    text = message.lower().strip()
    include, exclude = _find_genres(text)
    titles = _find_titles(message, data)
    year_from, year_to = _find_years(text)

    uid = None
    # 1-5 digits so an out-of-range id is *parsed* and then reported, rather than
    # silently failing to match and looking like an unintelligible request.
    if m := re.search(r"\buser\s*(?:id\s*)?#?\s*(\d{1,5})\b", text) or re.search(r"\bi'?m\s+(\d{1,5})\b", text):
        uid = int(m.group(1))

    u = Understanding(intent="unknown", user_id=uid, movies=titles,
                      include_genres=include, exclude_genres=exclude,
                      year_from=year_from, year_to=year_to, query=_clean_query(message))

    has = lambda *ws: any(w in text for w in ws)                      # noqa: E731
    peerish = has("similar taste", "people like me", "users like me", "similar users",
                  "people with similar", "others like me", "taste like mine", "similar to me")
    whyish = text.startswith("why") or has("why would i", "why do you think", "how come",
                                           "what makes you think", "justify", "explain")
    blindish = has("blind spot", "blindspot", "missing", "haven't watched", "havent watched",
                   "never watched", "not exploring", "under-explored", "genres am i missing",
                   "what am i not")
    profileish = has("my taste", "my profile", "what do i like", "describe my", "my rating history",
                     "what kind of movies do i")
    recommendish = has("what should i watch", "recommend", "suggest", "what else", "anything good",
                       "what to watch", "watch tonight", "something to watch", "give me a movie",
                       "any ideas", "what next", "more like")
    aboutish = has("tell me about", "what is", "what's", "plot of", "who is in", "about the movie")
    similarish = has("movies like", "films like", "similar to", "more like", "same vibe")
    likedish = re.search(r"\b(?:i liked|i loved|i enjoyed|i'm a fan of|im a fan of|i like)\b", text)

    # Pronoun reference: "why would I like *that*?" -> the last film discussed.
    if not titles and any(re.search(rf"\b{p}\b", text) for p in PRONOUNS):
        if state.last_movie:
            u.movies = [state.last_movie]
            u.used_pronoun = True
            u.notes.append(f"resolved pronoun to {state.last_movie!r} from the previous turn")

    if likedish and u.movies:
        u.liked_seed = list(u.movies)

    # A film the user named that we could not resolve. Worth detecting explicitly:
    # this catalogue is a *filtered* MovieLens subset, so real films (The Matrix,
    # Ocean's Eleven) are genuinely absent and the honest answer is to say so rather
    # than to quietly answer a neighbouring question.
    if not u.movies and (peerish or aboutish or similarish or likedish):
        # Only look for an unresolved title when the sentence is actually *about* a film.
        # Without this guard "put something on for me" parses "for me" as a title.
        # Take the *last* "about/of/like X" in the sentence: in "what do people like me
        # think about The Matrix" the first one ("like me") is not the film.
        tail = message.strip(" ?.!")
        lowered = tail.lower()
        best = None
        for cue in ("think about", "think of", "about", "of", "like"):
            idx = lowered.rfind(cue + " ")
            if idx < 0 or (idx and lowered[idx - 1] != " "):
                continue
            candidate = tail[idx + len(cue):].strip(" ?.!\"'")
            if 2 <= len(candidate) <= 45 and not _is_generic_phrase(candidate):
                if best is None or idx > best[0]:
                    best = (idx, candidate)
        if best:
            u.unresolved_title = best[1]

    # Order matters: the most specific evidence wins.
    if uid is not None and not (peerish or whyish or blindish or recommendish or similarish or aboutish) \
            and len(text.split()) <= 8:
        u.intent = "identify"
    elif peerish and u.movies:
        u.intent = "peer_opinion"
    elif peerish and u.unresolved_title:
        u.intent = "unresolved_movie"
    elif whyish and (u.movies or state.last_recommendations):
        u.intent = "explain"
    elif blindish:
        u.intent = "blind_spots"
    elif profileish:
        u.intent = "profile"
    elif peerish:
        u.intent = "similar_users"
    elif likedish and (exclude or recommendish):
        u.intent = "recommend"          # "I liked Toy Story but no more animation - what else?"
    elif similarish and u.movies:
        u.intent = "similar_movies"
    elif aboutish and u.movies:
        u.intent = "movie_info"
    elif recommendish and not u.query.strip(" ?"):
        u.intent = "recommend"
    elif recommendish and (include or exclude) and len(u.query.split()) <= 6:
        u.intent = "recommend"
    elif _looks_descriptive(text):
        u.intent = "content_search"
    elif recommendish:
        u.intent = "recommend"
    elif u.movies:
        u.intent = "movie_info"
    elif u.unresolved_title:
        u.intent = "unresolved_movie"
    else:
        # No recognisable signal. Earlier versions fell back to "recommend", which
        # meant nonsense input ("banana helicopter") got a confident list of films.
        u.intent = "unknown"
    return u


_GENERIC_PHRASES = {
    "me", "mine", "it", "that", "this", "movies", "films", "something", "anything",
    "my taste", "similar taste", "my taste to mine", "taste to mine", "tonight",
}


def _is_generic_phrase(text: str) -> bool:
    t = text.lower().strip()
    return t in _GENERIC_PHRASES or len(t) < 3 or t.split()[0] in {"my", "me", "mine"}


_DESCRIPTIVE_HINTS = [
    "mood", "vibe", "about a", "about an", "story", "movie where", "film where", "with a twist",
    "dark", "funny", "slow", "feel good", "feel-good", "mind bending", "mind-bending",
    "psychological", "gritty", "uplifting", "tearjerker", "atmospheric", "set in", "based on",
]


def _looks_descriptive(text: str) -> bool:
    """Free-text mood/theme request - route to content search rather than pure CF."""
    if any(h in text for h in _DESCRIPTIVE_HINTS):
        return True
    return bool(re.match(r"^(i want|i'd like|find me|show me|something|anything|give me)\b", text)) \
        and len(text.split()) >= 5
