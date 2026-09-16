# 04 edge cases

*Deliberate edge cases: no user id, a film that is not in this filtered catalogue, a nonexistent user, and an unparseable request.*

---

**you>** What should I watch tonight?

<details><summary>tool calls</summary>

```
(none)
```
</details>

```text
Tell me your user id first (for example "I'm user 30") - without it I'd just be reciting popular films rather than reading your history.
```

---

**you>** I'm user 4242

<details><summary>tool calls</summary>

```
(none)
```
</details>

```text
There's no user 4242 in this dataset - ids run from 1 to 610. Give me one in that range and I'll read their history.
```

---

**you>** I'm user 1

<details><summary>tool calls</summary>

```
get_user_profile(user_id=1)  [8 ms]
```
</details>

```text
You're user 1: 190 ratings, average 4.33 (a generous rater, +0.83 vs the dataset mean), spread 0.78.
Most watched: Action x74, Adventure x68, Comedy x68, Drama x54, Thriller x47.
Rated highest relative to your own baseline: Musical (4.74 over 19), Animation (4.56 over 18), Drama (4.52 over 54), Children (4.41 over 29).
Your top films: Canadian Bacon (1995) 5.0; Tombstone (1993) 5.0; Henry V (1989) 5.0; Rob Roy (1995) 5.0; Good Morning, Vietnam (1987) 5.0. Median release year in your history is 1990 (48% pre-1990).

Ask me what to watch, or about a specific film.
```

---

**you>** What do people like me think about The Matrix?

<details><summary>tool calls</summary>

```
(none)
```
</details>

```text
I can't find a film called 'The Matrix' in this catalogue. This is a filtered MovieLens subset of 5,135 films; some well-known titles were dropped upstream because they had no plot text.
Nothing in the catalogue has a similar title either.
Name one of those and I'll check what users with your taste made of it.
```

---

**you>** banana helicopter

<details><summary>tool calls</summary>

```
(none)
```
</details>

```text
I didn't get a clear question out of that, and I'd rather say so than guess.

I look things up in the ratings and plot data rather than guessing. Try:
  - "I'm user 30"
  - "What should I watch tonight?"
  - "I want a dark psychological thriller with a twist"
  - "What do people with similar taste to mine think about Pulp Fiction?"
  - "Why do you think I'd like that?"
  - "What's my blind spot?"
```

---

**you>** I want a feel good comedy about friendship

<details><summary>tool calls</summary>

```
search_movies(query='feel good comedy about friendship', n=6, include_genres=['Comedy'], min_ratings=10, for_user=1)  [486 ms]
```
</details>

```text
Searching plots, genres and tags for 'feel good comedy about friendship', limited to Comedy:

1. Brothers McMullen, The (1995) - Comedy
   predicted 4.2/5 for you - audience 3.33 from 12 ratings - plot match 0.50
   matched on: comedy, pat, barry, mcmullen, ann

2. Bring It On (2000) - Comedy
   predicted 4.1/5 for you - audience 2.69 from 34 ratings - plot match 0.50
   matched on: comedy, torrance, clovers, isis, aaron

3. Yes Man (2008) - Comedy
   predicted 4.6/5 for you - audience 3.62 from 34 ratings - plot match 0.50
   matched on: comedy, carl, allison, covenant, terrence

4. Bedazzled (2000) - Comedy
   predicted 4.2/5 for you - audience 3.07 from 27 ratings - plot match 0.49
   matched on: comedy, elliot, satan, alison, big mac

5. Clerks II (2006) - Comedy
   predicted 4.7/5 for you - audience 3.88 from 13 ratings - plot match 0.49
   matched on: comedy, randal, dante, quick stop, silent bob

6. Orange County (2002) - Comedy
   predicted 4.2/5 for you - audience 2.63 from 15 ratings - plot match 0.49
   matched on: comedy, shaun, stanford, admissions, orange county
```
