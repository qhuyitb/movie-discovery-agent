"""Interactive chat front-end.

    python -m movieagent.cli --user 1
    python -m movieagent.cli --user 1 --show-trace
    python -m movieagent.cli --backend anthropic --model claude-sonnet-5 --user 15

`--show-trace` prints the tool calls behind each answer, which is the honest way
to read this system: the prose is only a rendering of those calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movieagent.agent import build_agent          # noqa: E402
from movieagent.tools import MovieTools           # noqa: E402

BANNER = """\
Movie assistant - grounded in 5,135 films, 74,064 ratings, 610 users.
Commands: /user <id>   /trace   /help   /quit
"""


def run_turn(agent, message: str, show_trace: bool) -> None:
    turn = agent.chat(message)
    if show_trace:
        print("\n  \033[2m[investigating]\033[0m")
        for line in turn.trace_lines():
            print(f"  \033[2m- {line}\033[0m")
        for note in turn.notes:
            print(f"  \033[2m- note: {note}\033[0m")
        print()
    print(turn.text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conversational movie discovery over MovieLens data.")
    ap.add_argument("--user", type=int, default=None, help="MovieLens user id (1-610)")
    ap.add_argument("--backend", default="heuristic", choices=["heuristic", "openai", "anthropic"])
    ap.add_argument("--model", default=None, help="model name for the LLM backends")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (Groq, Ollama, ...)")
    ap.add_argument("--show-trace", action="store_true", help="print the tool calls behind each answer")
    ap.add_argument("--ask", action="append", default=None,
                    help="run one question and exit; repeatable for a scripted conversation")
    args = ap.parse_args()

    print("loading data and building the index (first run takes ~20s)...", file=sys.stderr)
    tools = MovieTools.bootstrap()
    agent = build_agent(args.backend, user_id=args.user, model=args.model,
                        base_url=args.base_url, tools=tools)

    if args.ask:
        for q in args.ask:
            print(f"\n\033[1m> {q}\033[0m\n")
            run_turn(agent, q, args.show_trace)
        return

    show_trace = args.show_trace
    print(BANNER)
    if args.user:
        run_turn(agent, f"I'm user {args.user}", show_trace)
    while True:
        try:
            msg = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/trace":
            show_trace = not show_trace
            print(f"trace {'on' if show_trace else 'off'}\n")
            continue
        if msg.startswith("/user"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].isdigit():
                agent.state.user_id = int(parts[1])
                run_turn(agent, f"I'm user {parts[1]}", show_trace)
            else:
                print("usage: /user 30\n")
            continue
        if msg == "/help":
            print(BANNER)
            continue
        run_turn(agent, msg, show_trace)


if __name__ == "__main__":
    main()
