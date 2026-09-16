# Movie discovery assistant - Quang Huy

A conversational assistant that answers questions about films by **investigating the
dataset**: profiling the user, finding people with similar taste, searching plot text,
and combining those signals into an answer that cites its own evidence.

Runs **fully offline with no API key**. An LLM-planned backend is included and enabled
with one flag if you want to run it with a key - the tools, the evidence and the
grounding rules are identical either way.

- **Report:** [`REPORT.md`](REPORT.md)
- **Sample conversations:** [`outputs/transcripts/`](outputs/transcripts/)
- **Evaluation results:** [`outputs/evaluation.md`](outputs/evaluation.md),
  [`outputs/component_eval.md`](outputs/component_eval.md),
  [`outputs/cohort_analysis.md`](outputs/cohort_analysis.md)

---

## Setup

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                     # deps, and puts `movieagent` on the path

python scripts/verify_dataset.py     # provided checker - confirms the data is in place
python scripts/smoke_test.py         # 23 end-to-end assertions, ~30s including index build
```

`requirements.txt` holds the same dependencies if you would rather not install the package;
in that case run the CLI as `PYTHONPATH=src python -m movieagent.cli ...`.

The first run builds a TF-IDF/LSA index over the 5,135 plot summaries (~20 s) and caches it
in `.cache/`. Later runs start in about a second.

## Use it

```bash
python -m movieagent.cli --user 1 --show-trace
```

`--show-trace` prints the tool calls behind each answer. That is the honest way to read
this system: the prose is only a rendering of those calls.

```
you> What do people with similar taste to mine think about Pulp Fiction?

  [investigating]
  - how_similar_users_rated(user_id=1, movie='Pulp Fiction', n_users=30)  [110 ms]

Pulp Fiction (1994) - Comedy|Crime|Drama|Thriller
25 of the users closest to your taste have rated it: average 4.64, which is +1.26
against their own baselines.
The whole dataset gives it 4.20 from 307 ratings, so your crowd rates it noticeably
higher than the crowd at large (+0.44).
You've already rated it 3.0.

Individual neighbours:
  - user 414 (similarity 0.097, 165 films in common) gave it 5.0 (+1.66 vs their average)
  ...
```

Useful commands inside the chat: `/user <id>`, `/trace`, `/help`, `/quit`.
Non-interactive: `python -m movieagent.cli --user 30 --ask "What's my blind spot?"`.

### With an LLM planner (optional)

```bash
pip install -e '.[llm]'        # or just: pip install openai
export ANTHROPIC_API_KEY=...
python -m movieagent.cli --user 1 --backend anthropic --model claude-sonnet-5 --show-trace

# any OpenAI-compatible endpoint works too
python -m movieagent.cli --backend openai --base-url http://localhost:11434/v1 --model llama3.1
```

Run against `gpt-4o-mini`; the output is committed as
[`outputs/transcripts/05_llm_backend.md`](outputs/transcripts/05_llm_backend.md) so this
path can be reviewed without a key of your own. Regenerate it with:

```bash
OPENAI_API_KEY=... python eval/run_llm_transcript.py --model gpt-4o-mini
```

Unlike the offline transcripts that one is not reproducible verbatim - the model is sampled
at temperature 0.3. It understands paraphrases the rule-based planner misses, but it plans
the tool calls less reliably; the report covers what it got right and wrong.

## Reproduce every number in the report

```bash
python eval/evaluate.py          # ranking + rating-prediction metrics   -> outputs/evaluation.{json,md}
python eval/tune_hybrid.py       # fusion-weight sweep                   -> outputs/hybrid_sweep.json
python eval/eval_components.py   # intent + content-search evaluation    -> outputs/component_eval.{json,md}
python eval/cohort_analysis.py   # why dense users score worse            -> outputs/cohort_analysis.md
python eval/run_transcripts.py   # sample conversations                  -> outputs/transcripts/*.md
OPENAI_API_KEY=... python eval/run_llm_transcript.py   # LLM backend     -> outputs/transcripts/05_llm_backend.md
```

The offline backend is deterministic, so `run_transcripts.py` reproduces the committed
transcripts verbatim. Total runtime is about two minutes on a laptop CPU.

## Layout

```
src/movieagent/
  config.py     paths and every tunable constant
  data.py       dataset loading, per-movie statistics, title resolution
  index.py      TF-IDF + LSA index over plots, genres and tags
  cf.py         collaborative filtering, user profiling, the hybrid ranker
  tools.py      the 10 investigation tools + their JSON schemas
  planner.py    rule-based language understanding (offline backend)
  nlg.py        renders tool evidence into English
  agent.py      the agent loop - heuristic and LLM backends
  cli.py        chat front-end
eval/           evaluation scripts (see above)
scripts/        verify_dataset.py (provided), smoke_test.py
outputs/        committed results: metrics and transcripts
```

## What it can answer

| Ask | What it actually does |
|---|---|
| "I'm user 30" | profiles the history: volume, generosity, genre affinities, era |
| "What should I watch tonight?" | hybrid ranking, then pulls the evidence behind the top picks |
| "I want a dark psychological thriller with a twist" | plot/genre/tag search, filtered and scored against the user |
| "What do people with similar taste think of Pulp Fiction?" | finds neighbours, looks up *their* ratings, compares to the crowd |
| "Why do you think I'd like that?" | the user's own ratings, plot matches, genre fit, peer ratings |
| "I liked Toy Story but I'm tired of animated movies" | seeds on Toy Story, hard-excludes Animation |
| "What's my blind spot?" | unexplored genres weighted by catalogue size, with entry points |
| "Who has taste like mine?" | nearest neighbours with overlap counts and their favourites |
