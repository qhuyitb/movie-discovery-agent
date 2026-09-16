#!/usr/bin/env python3
"""Fast end-to-end check that the system is wired up correctly.

Not a substitute for the evaluation - it asserts plumbing and grounding
invariants, not recommendation quality.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movieagent.agent import build_agent      # noqa: E402
from movieagent.tools import TOOL_NAMES, MovieTools   # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  - ' + detail if detail and not condition else ''}")
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("bootstrapping...")
    tools = MovieTools.bootstrap()

    print("\ndata")
    check("5,135 movies loaded", len(tools.data.movies) == 5135, str(len(tools.data.movies)))
    check("74,064 ratings loaded", len(tools.data.ratings) == 74064, str(len(tools.data.ratings)))
    check("610 users", tools.data.ratings["userId"].nunique() == 610)
    check("title resolution handles 'Godfather'", tools.data.resolve_movie("godfather") is not None)
    check("absent film resolves to None", tools.data.resolve_movie("The Matrix") is None)

    print("\ntools")
    sample_args = {
        "search_movies": {"query": "heist film", "n": 3},
        "get_movie_details": {"movie": "Pulp Fiction"},
        "find_similar_movies": {"movie": "Pulp Fiction"},
        "how_similar_users_rated": {"user_id": 1, "movie": "Pulp Fiction"},
        "explain_recommendation": {"user_id": 1, "movie": "Pulp Fiction"},
    }
    for name in sorted(TOOL_NAMES):
        result = tools.call(name, sample_args.get(name, {"user_id": 1}))
        check(f"{name} returns evidence", isinstance(result, dict) and "error" not in result,
              str(result)[:120])

    print("\nagent (offline backend)")
    agent = build_agent("heuristic", tools=tools)
    turn = agent.chat("What should I watch tonight?")
    check("refuses to recommend without a user id", "user id" in turn.text.lower())

    agent.chat("I'm user 1")
    turn = agent.chat("What should I watch tonight?")
    check("recommends once identified", turn.intent == "recommend" and "predicted" in turn.text)
    check("every answer is backed by tool calls", len(turn.trace) > 0, "empty trace")

    turn = agent.chat("Why do you think I'd like that?")
    check("resolves 'that' to the last recommendation", turn.intent == "explain")

    turn = agent.chat("What do people like me think about The Matrix?")
    check("admits when a film is not in the catalogue", turn.intent == "unresolved_movie")

    turn = agent.chat("banana helicopter")
    check("says it does not understand rather than guessing", turn.intent == "unknown")

    turn = agent.chat("I'm user 4242")
    check("rejects an out-of-range user id", "no user 4242" in turn.text)
    check("keeps the previous valid user", agent.state.user_id == 1)

    print(f"\n{'all checks passed' if not FAILURES else str(len(FAILURES)) + ' FAILED: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
