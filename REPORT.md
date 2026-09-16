# Report: Quang Huy

*System: a conversational movie-discovery assistant over the filtered MovieLens subset.
Code and setup in [`README_SOLUTION.md`](README_SOLUTION.md); every number below is
reproduced by the scripts in `eval/`.*

---

## Problem Analysis

**Who the users are.** Someone who already knows what a search box does and has not
found it enough. They come with one of four shapes of question, and they are not the same
problem:

| Shape | Example | What it actually needs |
|---|---|---|
| Open-ended | "what should I watch tonight?" | a ranked model of *this* person's taste |
| Constrained mood | "dark psychological thriller with a twist" | retrieval over content + a quality filter |
| Multi-hop social | "what do people like me think of Pulp Fiction?" | neighbours -> their ratings of X -> synthesis |
| Introspective | "why would I like that?", "what's my blind spot?" | evidence *about the user*, not about films |

The third and fourth are the interesting ones, and they are what separate an assistant
from a search engine: neither can be answered by retrieving a document. They require
looking up one thing to decide what to look up next.

**What makes a recommendation good in conversation.** Three things that offline ranking
metrics do not measure:

1. **It is defensible.** In chat the user can always ask "why?", and the honest answer has
   to come from their history - "14 of your closest neighbours rated it 4.36, +0.88 against
   their own averages" - not from the recommender's confidence score. A system that cannot
   produce that sentence has not really explained anything.
2. **It is calibrated.** Saying "5.0 for you" on the strength of one stranger's rating is
   worse than saying "thin evidence". 51% of this catalogue has fewer than 5 ratings, so
   most candidates come with very little support behind them.
3. **It obeys constraints.** "I'm tired of animated movies" is a hard filter, not a soft
   preference, and getting that wrong destroys trust faster than a mediocre ranking.

**The key technical challenges.**

- *Sparsity is on the movie side, not the user side.* Users average 121 ratings; more than
  half the films have under 5. So user-based CF has plenty to work with while item-based CF
  and any popularity signal are starved - which is exactly what the evaluation shows.
- *Grounding.* An LLM asked about Pulp Fiction will answer from its own memory of Pulp
  Fiction. That answer is not about this dataset, and the requirement is explicitly that
  explanations come from historical data.
- *Multi-hop orchestration.* The interesting questions need a plan, and the plan depends on
  results that do not exist until the first tool has run.
- *Evaluating a conversation.* There is no held-out label for "was that a good explanation".
  Ranking metrics measure one component of four.

---

## Approach

### Breakdown

I split the system along the line where the failure modes differ, so each piece could be
evaluated separately:

```
data.py / index.py   the dataset: stats, title resolution, TF-IDF + LSA over plots
cf.py                the models: user-CF, item-CF, content, and the hybrid fusion
tools.py             10 investigation tools - the only things that may touch the data
planner.py           language understanding: intent + slots + pronouns   (offline backend)
agent.py             the loop: plan -> call tools -> narrate, over either backend
nlg.py               renders tool evidence into English
```

**The one rule that shapes everything: tools return evidence, never prose.** Every tool
hands back a JSON-able dict of numbers and titles pulled from the dataset, and the narration
layer is only allowed to rephrase what a tool returned. Every call is recorded in a trace
that `--show-trace` prints. This is what makes "explains its reasoning using historical
data" checkable rather than a claim - you can read the answer, then read the four tool calls
underneath it, and see that nothing was invented in between.

The ten tools are deliberately *investigative*, not CRUD: `get_user_profile`,
`get_user_ratings`, `search_movies`, `recommend_for_user`, `find_similar_users`,
`how_similar_users_rated`, `explain_recommendation`, `find_blind_spots`,
`get_movie_details`, `find_similar_movies`. "What do people like me think of Pulp Fiction?"
is one call because the multi-hop join (neighbours -> their ratings -> compare to the crowd)
belongs in the data layer where it can be tested, not in a prompt where it can be forgotten.

### Methods

**Retrieval.** TF-IDF (word + bigram) blended with a 256-dim LSA view of the same matrix,
over plot text plus genres and tags. TF-IDF catches concrete plot nouns ("heist", "amnesia");
LSA rescues vibe queries where no single word has to match.

**Collaborative filtering.** Significance-weighted cosine on mean-centred ratings, with
shrinkage `sim * n_co/(n_co + 25)` so a 1.0 similarity over 3 shared films does not
outvote a 0.3 over 150. The matrix is 610 x 5,135 - dense float32, no sparse acrobatics,
which keeps `cf.py` readable.

**Fusion.** Min-max normalise the three scorers over the *candidate set* (so the weights
mean something), blend, then add a small Bayesian-damped popularity prior as a tie-breaker.

**Narration.** Rule-based by default, LLM optional (`--backend anthropic|openai`) behind
the same tool schemas and the same grounding rules.

### Alternatives rejected

| Considered | Why not |
|---|---|
| Sentence-transformer / OpenAI embeddings for plots | Better semantics, but adds a GPU or an API dependency to a system whose main weakness is elsewhere. TF-IDF+LSA scored genre P@10 = 0.99 on the probe queries - the content component is not what is broken. |
| Matrix factorisation (ALS / SVD++) | The standard answer, and it would likely beat neighbourhood CF. But it destroys the thing the brief actually asks for: "12 of your closest neighbours rated it 4.54" is a sentence; "latent factor 7 is high" is not. I chose the model I could explain to the user. |
| Free-form LLM agent as the only backend | Cannot be run or reproduced without a key, and invites exactly the ungrounded answer the brief warns about. |
| Building on tags | 2,440 tags over 5,135 films. Used as a retrieval signal where present, never depended on. |

### Decision Log

| Decision | Alternative considered | Why I chose this |
|---|---|---|
| **Tools return evidence, prose is a rendering of it.** The narration layer may only rephrase tool output; every claim is traceable to a recorded call. | Let the model answer freely and ask it nicely to be grounded. | It turns grounding from a promise into a property you can audit with `--show-trace`. It also means the offline and LLM backends make the *same* factual claims - only the wording differs. Cost: the assistant cannot say anything the tool layer has no tool for. |
| **Rule-based planner as the default backend; LLM behind a flag.** | Require an API key and use function calling for everything. | I had no key while building, and a deterministic backend makes the committed transcripts reproducible verbatim - a reviewer can check my evidence without running anything. The cost is measured, not hand-waved: intent accuracy is 1.00 on phrasings the rules target and **0.48 on paraphrases they do not**. The LLM backend replaces `planner.py` and nothing else. I have since run it (`outputs/transcripts/05_llm_backend.md`) and the trade is not one-sided - see the LLM backend section below. |
| **Hybrid fusion (0.60 user-CF / 0.20 item-CF / 0.20 content, no popularity prior) instead of pure user-CF, and coverage as a tie-breaker between near-equal weightings.** | Pure user-CF (NDCG@10 0.067), and the sweep's top NDCG row, 0.80/0.10/0.10 + 0.1 prior, at 0.0741. | Against pure user-CF the accuracy gain is small (0.067 -> 0.073) but the behavioural gain is not: catalogue coverage goes 5.6% -> 10.2% and long-tail share 12.8% -> 23.3%, which is the whole point of a discovery assistant. Against the sweep winner I gave up 0.0011 NDCG to nearly double coverage (0.055 -> 0.102) - on a surface this flat (NDCG spans .067-.074 across every sensible mix) that trade is free in practice. I ranked the sweep by NDCG but did not pick by it. |

---

## Evaluation

I evaluated the system as four components, because they fail for different reasons and a
single number would hide all of it.

**Protocol** (`eval/evaluate.py`): temporal leave-last-out per user - holdout is each user's
last 20% of ratings, capped at 10, users with >=20 ratings. Relevant = held-out rating >= 4.0.
Candidates = unseen films with >=5 *training* ratings. Popularity and item statistics are
recomputed from the training half only, so the test set does not leak into the candidate
filter. 69,427 train / 4,637 test ratings over 533 users.

### 1. Ranking quality (@10)

| method | recall | precision | NDCG | MAP | hit-rate | catalogue coverage | long-tail share |
|---|---|---|---|---|---|---|---|
| `popularity` | 0.051 | 0.025 | 0.045 | 0.023 | 0.189 | 0.012 | 0.089 |
| `user_cf` | 0.080 | 0.038 | 0.067 | 0.034 | 0.296 | 0.056 | 0.128 |
| `item_cf` | 0.024 | 0.014 | 0.021 | 0.008 | 0.117 | 0.289 | 0.655 |
| `content` | 0.012 | 0.007 | 0.012 | 0.005 | 0.062 | 0.183 | 0.843 |
| `hybrid` | **0.086** | **0.042** | **0.073** | **0.036** | **0.328** | 0.102 | 0.233 |

The hybrid beats the popularity baseline by 68% on recall and hits at least one liked film
for a third of users, but recall@10 of 0.086 means it finds under a tenth of what the user
went on to enjoy. That is normal for a dataset this size, and it is still not a system I
would ship on these numbers alone.

The ordering is the informative part. **Content-based ranking is the worst recommender and
the best explainer** - 0.012 recall, but 84% long-tail coverage and the only signal that can
say *why* a film resembles what you loved. Item-CF is similar: nearly useless for ranking
here (0.024) because the item side is where the sparsity is, yet it produces the "people who
liked this also liked Goodfellas, which you gave 5.0" line that makes an explanation land.
I keep both at low weight because of what they contribute to the explanations, not because
of what they do to the metric.

**Rating prediction** (secondary check, same split): item-CF RMSE 0.898, user-CF 0.912,
against a per-user-mean baseline of 0.973 and a global mean of 1.066. Both models beat the
baselines; the gap is modest, which is the same story as above.

### 2. Where it works and where it fails, by user

| method | dense (>150) n=112 | medium (50-150) n=160 | sparse (<50) n=225 |
|---|---|---|---|
| `popularity` | 0.035 / 0.032 | 0.049 / 0.041 | 0.061 / 0.055 |
| `user_cf` | 0.041 / 0.036 | 0.084 / 0.072 | 0.097 / 0.079 |
| `hybrid` | 0.036 / 0.034 | 0.090 / 0.081 | **0.109 / 0.087** |

*(recall@10 / NDCG@10)*

This is the most interesting result in the evaluation and it runs backwards: the system is
**worst on the users it knows most about**. Recall falls from 0.109 on sparse users to 0.036
on dense ones - and note that re-weighting the hybrid toward coverage *widened* that gap
(dense went 0.039 -> 0.036 while sparse went 0.108 -> 0.109), so the fix is not in the weights.
Cold start is not the failure mode here - *warm* start is.

`eval/cohort_analysis.py` explains it without involving the model at all
(`outputs/cohort_analysis.md`):

| cohort | median train ratings of their held-out favourites | share that are long-tail | share of the 200 most-rated films already seen |
|---|---|---|---|
| dense (>150) | 35 | 67.5% | **48.4%** |
| medium (50-150) | 42 | 56.6% | 19.3% |
| sparse (<50) | 66 | 38.5% | 8.1% |

A dense user has already watched half of everything the ratings matrix knows well, so the
films they go on to enjoy are ones almost nobody here has rated - 12% of their held-out
favourites are not even in the candidate pool. They are being asked a strictly harder
question, and collaborative filtering has least to say exactly there. My ranker does not
degrade gracefully into content-based discovery when the CF signal runs out; it keeps
weighting user-CF at 0.60 for everybody and returns the tail of the head.

### 3. Intent classification (`eval/eval_components.py`)

35 hand-written utterances. **Overall accuracy 0.69 - 1.00 on phrasings the rules target,
0.48 on paraphrases they do not.** Representative misses:

| utterance | expected | got |
|---|---|---|
| surprise me | `recommend` | `unknown` |
| anything worth watching? | `recommend` | `unknown` |
| how do folks with my kind of taste rate Fargo? | `peer_opinion` | `movie_info` |
| on what basis? | `explain` | `unknown` |
| am I stuck in a rut genre wise | `blind_spots` | `unknown` |
| what else is like Memento | `similar_movies` | `recommend` |

This is the honest price of a rule-based planner and the single strongest argument for the
LLM backend. Note the failures are mostly `unknown` rather than wrong actions - an earlier
version fell back to `recommend`, which produced confident answers to questions nobody
asked, so I made the fallback admit it did not understand.

### 4. Content search

10 mood/theme queries with hand-labelled target genres: **mean genre precision@10 = 0.99**.
Retrieval is not the problem. Quality is: returned films average **3.37** against a dataset
mean of 3.50, and 63% have fewer than 10 ratings. Content search is quality-blind by
construction - see failure 1.

### 5. The LLM backend

I ran the optional backend against `gpt-4o-mini` after the rest of this report was written
(`eval/run_llm_transcript.py`, output in
[`outputs/transcripts/05_llm_backend.md`](outputs/transcripts/05_llm_backend.md)). Four
turns, two of them deliberately baited. What it changes:

**Grounding held.** Asked *"Is Pulp Fiction better than The Godfather? What did critics say
at the time?"* it looked up both films and answered from the neighbour data alone. It
invented no critic. Asked to describe the plot of The Matrix - which is not in this
catalogue - it made five tool calls, found nothing, and said so. That was the failure mode I
was most worried about when I wrote the system prompt, and on these four turns it did not
occur. Four turns is not a grounding evaluation, but it is better than the zero turns I had
before.

**Planning got worse on the case that needs planning.** For *"I liked Toy Story but I'm
tired of animated movies"* the rule-based planner emits
`recommend_for_user(user_id=30, exclude_genres=['Animation'], similar_to=['Toy Story'])`.
The model got the hard constraint right both times I ran it and **dropped the seed both
times** - once calling the tool with no `similar_to` at all, once substituting
`include_genres` copied from the user's profile. So it answers "what else should I watch,
no animation" rather than the question actually asked, which was anchored on Toy Story.
The model is also noticeably less efficient: five tool calls to establish that The Matrix is
absent, where the rules resolve the title once and stop.

The honest summary is that the two backends fail in opposite directions. The rules
understand fewer sentences but execute the ones they understand exactly; the model
understands almost anything and executes it approximately. Given more time I would keep the
model for intent and slot extraction and keep the rules for turning a resolved intent into a
tool call, which is the part the model has no advantage at.

### 6. Qualitative

Four transcripts in [`outputs/transcripts/`](outputs/transcripts/), reproducible verbatim
via `eval/run_transcripts.py`, covering the dense user, the sci-fi user, the cold-start user
with a hard constraint, and deliberate edge cases (no user id, user 4242, a film not in the
filtered catalogue, and "banana helicopter"). The multi-hop and constraint cases work: user
30's "I liked Toy Story but I'm tired of animated movies" correctly seeds on Toy Story and
hard-excludes Animation; the Pulp Fiction question correctly reports +0.44 against the crowd
*and* volunteers that the user already rated it 3.0.

---

### Failure Analysis

#### Failure 1 - Content search happily recommends bad films

**Asked** (user 1): *"I want a feel good comedy about friendship"*

**Returned:** Brothers McMullen (audience 3.33), **Bring It On (2.69)**, Yes Man (3.62),
Bedazzled (3.07), Clerks II (3.88), **Orange County (2.63)**.

**Why it failed.** Two compounding problems. First, ranking is by plot similarity alone -
audience rating is *displayed* but does not enter the score, so a well-matched bad film
outranks a slightly-less-matched great one. Second, look at what the matches are made of:
`matched on: comedy, pat, barry, mcmullen, ann` - `comedy, shaun, stanford, admissions`.
TF-IDF is latching onto character names from the plot synopsis, because proper nouns are
exactly the rare high-IDF tokens it loves. Nothing in that list encodes "feel good" or
"friendship". The genre label is doing all the honest work, which is why genre P@10 is 0.99
while the films are mediocre: the metric I chose measures the half that works.

**Fix.** Blend the Bayesian-damped quality prior into content ranking rather than only
displaying it (the machinery already exists in `cf.py`); strip person-name tokens with an
NER pass or a document-frequency floor over the character-name vocabulary; and weight tags
above plot text, since "feel-good" is the kind of thing a tagger writes down and a plot
summary never states.

#### Failure 2 - The recommender is worst for the best-known users

**Asked** (user 1, 190 ratings): *"What should I watch tonight?"*

**Returned:** The Great Escape (1963), Yojimbo (1961), The Godfather (1972), Bridge on the
River Kwai (1957), Wallace & Gromit (1993).

**Why it is a failure.** These are not wrong - they are canon, and the neighbour evidence is
real (12 neighbours at 4.54, +1.20 over their baselines). But for a user with 190 ratings
who has already seen 48% of this dataset's popular head, "have you tried The Godfather" is
close to useless as *discovery*. The cohort analysis above quantifies it: recall 0.036 for
this group against 0.109 for sparse users. The system is confidently retrieving the most
famous films the user has not happened to rate. Item 4 and item 5 do not even carry a real
explanation - they fall back to "backed by similar-user signal (support 0.9)", which is the
system admitting it has neighbours but no story.

There is a second, cosmetic-looking bug underneath it: three of the five predictions are
5.0 *capped*. User 1 averages 4.33, so mean-centred CF routinely predicts above the ceiling
and the top of the list has no usable ordering left. I surface this in the answer ("use the
ordering and the evidence rather than the absolute number") instead of hiding it, but
surfacing a broken number is not the same as fixing it.

**Fix.** Make the fusion weights a function of how much CF signal actually survives the
candidate filter - when a user has exhausted the well-rated catalogue, shift weight to
content and item-CF, which have 84% and 66% long-tail coverage respectively. Add an explicit
novelty/serendipity term (penalise items by training popularity) and report a
serendipity metric alongside recall, since recall@10 actively rewards the behaviour I am
calling a failure. For the ceiling, fit a per-user monotone calibration to the 0.5-5.0 range
instead of clipping.

#### Failure 3 - The planner does not understand paraphrases

**Asked:** *"surprise me"* -> `unknown` (falls back to the help text).
**Asked:** *"how do folks with my kind of taste rate Fargo?"* -> classified `movie_info`,
so it answers with Fargo's audience rating and plot instead of running the neighbour
lookup - the exact multi-hop question the brief singles out, missed on phrasing alone.

**Why it failed.** `planner.py` is keyword rules. "surprise me" contains no recommendation
cue in the list; "folks with my kind of taste" does not match the `similar taste` /
`people like me` patterns, while the title `Fargo` does match, so the movie-info rule wins.
Accuracy on paraphrases outside the targeted phrasings is 0.48.

**Fix.** This is what `--backend anthropic` is for - it swaps exactly this file and reuses
the same tools and grounding rules. Cheaper interim fixes: score intents rather than
first-match on rules (so a matched title cannot silently outrank a peer-opinion cue), and
embed the 35 eval utterances as a nearest-neighbour intent classifier, which needs no API
key and would catch most paraphrase misses.

#### An honesty note that belongs here

The weights in this report are not the ones I built the system with. I originally fixed
`0.55/0.25/0.20 + 0.10` in `config.py` *before* the final sweep ran, and only noticed while
writing this report that `outputs/hybrid_sweep.json` had `0.60/0.20/0.20` with no popularity
prior dominating it on **both** axes - NDCG 0.0730 / coverage 0.1019 against 0.0705 / 0.0900.
I changed the weights and re-ran every script, so all the numbers above are from the corrected
configuration.

Two things worth saying about that. First, the gap sat entirely inside the plateau, and
switching changed no conclusion in this report - which is the useful evidence that the
plateau claim was real rather than a hedge. Second, it made failure 2 slightly *worse*: dense
users went 0.039 -> 0.036 while sparse users went 0.108 -> 0.109. Tuning the fusion is not a
lever on the problem I care most about. That is more useful to know than the 0.0025 NDCG
the switch gained me.

---

## Reflection

**What works well.**

- **The grounding architecture.** Tools return evidence, prose renders it, every call is
  traced. I can hand a reviewer `--show-trace` and let them audit any sentence the system
  produced. Swapping the rule-based planner for a real LLM changes the wording and the
  intent accuracy, not the facts.
- **Multi-hop questions actually multi-hop.** "What do people like me think of X" resolves
  neighbours, looks up *their* ratings, compares against their own baselines and against the
  crowd, and mentions the user's own rating if it exists. The +0.44-vs-crowd framing is the
  answer the question was really asking for.
- **Calibrated honesty.** Thin evidence is labelled ("3 ratings - thin evidence"), films
  outside the filtered catalogue get a specific explanation rather than a hallucinated plot,
  unparseable input gets "I didn't get a clear question out of that" instead of a guess, and
  the blind-spot tool refuses to call a genre a gap when the user has sampled it and rated
  it low.
- **Evaluation that names its own weak spots** instead of reporting one flattering number.

**What doesn't work well.**

- **The planner** - 0.48 on paraphrases. Anyone who does not phrase things the way I did
  hits the help text. This is the first thing a real user would notice.
- **Discovery for engaged users.** Failure 2. The people most invested in the system get the
  least out of it, and the architecture has no mechanism to notice that the CF signal has
  run out.
- **Content search is quality-blind**, and its similarity is partly driven by character
  names.
- **Predicted ratings are not calibrated** - generous raters pile up at a clipped 5.0.
- **The LLM backend plans worse than the rules it replaces.** It fixes the paraphrase
  problem and it stayed grounded under pressure, but on the one case where the offline
  planner extracts a seed film it dropped the seed both times I ran it. See below.
- **Absolute accuracy is low.** Recall@10 = 0.086. Honest framing: this system is better at
  explaining a recommendation than at making one.

**With more time or resources.**

1. **Fix the metric before the model.** Recall@10 rewards recommending famous films, which
   is the behaviour I diagnosed as failure 2. I would add serendipity and novelty-weighted
   recall, then re-tune - otherwise every optimisation makes the product worse while the
   number goes up.
2. **Adaptive fusion**, with weights conditioned on surviving CF support per user.
3. **A small LLM-judge evaluation of the explanations**, which is the component I claim as
   the system's strength and the only one I currently evaluate by reading transcripts.
4. **Matrix factorisation as a scoring layer behind the neighbourhood explanations** - use
   ALS to rank, keep neighbour evidence to explain. That is probably the right architecture
   and I avoided it only because I wanted rank and explanation to come from the same object.
5. **Actually run the LLM backend** and measure whether it stays grounded under adversarial
   questions ("is this better than The Godfather?" invites the model's own film knowledge).

---

## Open Section

**The sparsity is on the wrong side, and that shaped everything.** The brief says it -
"users are dense, movies are sparse" - but the consequence only became clear in the
per-method table. Item-CF at 0.024 recall and content at 0.012 look like broken components.
They are not; they are the only components with anything to say about 84% and 66% of the
catalogue respectively. A single accuracy number would have told me to delete both, and
deleting both would have removed every "people who liked this also liked..." and "this
resembles a film you rated 5.0" line in the system - the entire explanation layer. The
components that rank worst are the ones that explain best, and I only found that out by
evaluating them separately.

**What I would tell a product owner.** I would not ship this as a recommender on its own.
It is more useful as an explanation layer on top of someone else's ranker: the multi-hop peer
lookup, the blind-spot analysis and the evidence-backed "why" are all in better shape than
the ranking they currently decorate. The blind-spot tool in particular gives a genuinely useful answer to a
question no search box can take - user 30 has 0 Romance, 0 Fantasy, 0 Musical, 0 Western
ratings, and the tool answers with entry points that have audience numbers attached.

**A small thing I liked.** The blind-spot tool weights an unexplored genre by how much of it
the catalogue actually holds, and refuses to call a genre a blind spot when the user has
sampled it and rated it below their own average. That distinction, gap versus preference,
took three lines of code and stops the tool from badgering people about genres they have
already decided they dislike.

**Time spent.** Roughly 2h on the system, 1h on evaluation, 1h on the report and cleanup.
Most of what I learned about the system came out of the evaluation hour, not the building.
