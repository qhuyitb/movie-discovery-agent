# 03 user30 cold start

*User 30 - only 18 ratings, average 4.61. The cold-start case, plus a hard constraint.*

---

**you>** I'm user 30

<details><summary>tool calls</summary>

```
get_user_profile(user_id=30)  [4 ms]
```
</details>

```text
You're user 30: 18 ratings, average 4.61 (a generous rater, +1.12 vs the dataset mean), spread 0.66.
Most watched: Action x15, Adventure x9, Sci-Fi x9, Crime x5, Drama x4.
Rated highest relative to your own baseline: Drama (5.00 over 4), IMAX (5.00 over 3).
Your top films: Braveheart (1995) 5.0; Up (2009) 5.0; Iron Man (2008) 5.0; Star Trek (2009) 5.0; Batman Begins (2005) 5.0. Median release year in your history is 1992 (44% pre-1990).

Ask me what to watch, or about a specific film.
```

---

**you>** I liked Toy Story but I'm tired of animated movies - what else?

<details><summary>tool calls</summary>

```
recommend_for_user(user_id=30, n=5, exclude_genres=['Animation'], similar_to=['Toy Story'])  [11 ms]
explain_recommendation(user_id=30, movie='Princess Bride, The')  [74 ms]
explain_recommendation(user_id=30, movie='Back to the Future')  [77 ms]
explain_recommendation(user_id=30, movie='Mary Poppins')  [74 ms]
get_user_profile(user_id=30)  [4 ms]
```
</details>

```text
Working from your 18 ratings (you average 4.61), steered toward Toy Story (1995), excluding Animation:

1. Princess Bride, The (1987) - Action|Adventure|Comedy|Fantasy|Romance
   predicted 5.0/5 for you - audience 4.23 from 142 ratings
   why: 14 of your closest neighbours rated it 4.36, +0.88 vs their own baseline; people who liked it also liked Star Wars: Episode V - The Empire Strikes Back (1980), which you gave 5.0; Adventure runs +0.17 above your average

2. Back to the Future (1985) - Adventure|Comedy|Sci-Fi
   predicted 5.0/5 (capped) for you - audience 4.04 from 171 ratings
   why: 16 of your closest neighbours rated it 4.47, +0.96 vs their own baseline; people who liked it also liked Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981), which you gave 5.0; Adventure runs +0.17 above your average

3. Mary Poppins (1964) - Children|Comedy|Fantasy|Musical
   predicted 4.9/5 for you - audience 3.89 from 71 ratings
   why: 7 of your closest neighbours rated it 4.07, +0.96 vs their own baseline; people who liked it also liked Up (2009), which you gave 5.0; Children runs +0.39 above your average

4. Pirates of the Caribbean: The Curse of the Black Pearl (2003) - Action|Adventure|Comedy|Fantasy
   predicted 4.9/5 for you - audience 3.78 from 149 ratings
   backed by similar-user signal (support 0.7)

5. Elf (2003) - Children|Comedy|Fantasy
   predicted 4.8/5 for you - audience 3.69 from 39 ratings
   backed by similar-user signal (support 0.2)

("capped" = the model predicted above 5.0 and I clipped it. You rate generously, so predicted scores bunch at the ceiling; use the ordering and the evidence rather than the absolute number.)
```

---

**you>** Why do you think I'd like that?

<details><summary>tool calls</summary>

```
explain_recommendation(user_id=30, movie='Princess Bride, The')  [74 ms]
note: resolved pronoun to 'Princess Bride, The' from the previous turn
```
</details>

```text
Princess Bride, The (1987) - Action|Adventure|Comedy|Fantasy|Romance - predicted 5.00/5 for you, audience 4.23 from 142 ratings.

From your own ratings (people who rated these the way you did tend to rate this film similarly):
  - Star Wars: Episode V - The Empire Strikes Back (1980) - you gave 5.0 (taste overlap 0.42)
  - Star Wars: Episode IV - A New Hope (1977) - you gave 5.0 (taste overlap 0.39)
  - Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) - you gave 5.0 (taste overlap 0.37)
  - Indiana Jones and the Last Crusade (1989) - you gave 5.0 (taste overlap 0.30)
  - Star Wars: Episode VI - Return of the Jedi (1983) - you gave 5.0 (taste overlap 0.29)

Films in your history with the most similar plots:
  - Indiana Jones and the Last Crusade (1989) - you gave 5.0 (plot similarity 0.29)
  - Iron Man (2008) - you gave 5.0 (plot similarity 0.28)
  - Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) - you gave 5.0 (plot similarity 0.20)

Users with taste closest to yours: 14 of them rated it, average 4.36 (+0.88 against their own averages). Everyone else: 4.23 from 142 ratings.

Genre fit versus your own average: Action -0.01 over 15 films, Adventure +0.17 over 9 films, Comedy -0.11 over 2 films.
Plot keywords: fingered, montoya, fantasy romance, adventure comedy, comedy fantasy, action adventure.
```

---

**you>** Who has taste like mine?

<details><summary>tool calls</summary>

```
find_similar_users(user_id=30, n=5)  [5 ms]
```
</details>

```text
Users whose rating patterns line up with user 30:
  - user 393: similarity 0.064 over 9 shared films, 80 ratings, averages 3.67
      favourites: Constantine (2005), Stardust (2007), Hellboy (2004)
  - user 407: similarity 0.052 over 7 shared films, 13 ratings, averages 3.92
      favourites: Iron Man (2008), Fight Club (1999), Minority Report (2002)
  - user 21: similarity 0.050 over 15 shared films, 238 ratings, averages 3.42
      favourites: Idiocracy (2006), Back to the Future Part III (1990), Back to the Future Part II (1989)
  - user 167: similarity 0.043 over 9 shared films, 136 ratings, averages 3.42
      favourites: Wallace & Gromit: A Close Shave (1995), Wallace & Gromit: The Wrong Trousers (1993), Dogma (1999)
  - user 166: similarity 0.037 over 9 shared films, 143 ratings, averages 4.07
      favourites: Garden State (2004), Aladdin (1992), Lost in Translation (2003)

(similarity = cosine on mean-centred ratings, shrunk by co-rated count.)
```

---

**you>** What's my blind spot?

<details><summary>tool calls</summary>

```
find_blind_spots(user_id=30)  [146 ms]
```
</details>

```text
Across your 18 ratings, you lean hard on Action (15 films), Adventure (9 films), Sci-Fi (9 films).

Barely touched:
  - Romance: 0 rated, 204 well-known titles available here
      start with: Princess Bride, The (4.23 from 142); Forrest Gump (4.16 from 329)
  - Fantasy: 0 rated, 117 well-known titles available here
      start with: Princess Bride, The (4.23 from 142); Pirates of the Caribbean: The Curse of the Black Pearl (3.78 from 149)
  - Musical: 0 rated, 47 well-known titles available here
      start with: Blues Brothers, The (3.81 from 84); Mary Poppins (3.89 from 71)
  - Western: 0 rated, 24 well-known titles available here
      start with: Dances with Wolves (3.84 from 164); 3:10 to Yuma (4.06 from 25)

Lightly sampled (and how you felt about them):
  - Thriller: 3 rated, you average 4.17 there (-0.44 vs your overall)
      try: Casino Royale; X-Men: First Class
  - Drama: 4 rated, you average 5.00 there (+0.39 vs your overall)
      try: American History X; Fight Club
  - Crime: 5 rated, you average 4.80 there (+0.19 vs your overall)
      try: American History X; Fight Club

Caveat: a gap is only a blind spot if you'd enjoy the genre. Where you have sampled a genre and rated it below your average, I read that as a preference, not a gap.
```
