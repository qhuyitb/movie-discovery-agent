# Component evaluation

## 1. Intent classification (offline planner)

35 hand-written utterances. **Overall accuracy 0.69** - 1.00 on phrasings the rules target, **0.48 on paraphrases they do not**. The gap is the honest cost of a rule-based planner.

| utterance | expected | got |
|---|---|---|
| my id is 42 | `identify` | `unknown` |
| put something on for me | `recommend` | `unknown` |
| give me a film for tonight | `recommend` | `content_search` |
| anything worth watching? | `recommend` | `unknown` |
| surprise me | `recommend` | `unknown` |
| how do folks with my kind of taste rate Fargo? | `peer_opinion` | `movie_info` |
| on what basis? | `explain` | `unknown` |
| which corners of cinema have I ignored | `blind_spots` | `unknown` |
| am I stuck in a rut genre wise | `blind_spots` | `unknown` |
| what else is like Memento | `similar_movies` | `recommend` |
| who rates films the way I do | `similar_users` | `movie_info` |

## 2. Content search

10 mood/theme queries with hand-labelled target genres. **Mean genre precision@10 = 0.99.**

Films returned average **3.37** against a dataset mean of 3.50, and 63% of them have fewer than 10 ratings - content search is quality-blind by construction.

| query | genre P@10 | avg audience rating | top 3 |
|---|---|---|---|
| dark psychological thriller with a twist | 1.00 | 3.69 | Game, The (1997), Memento (2000), Fight Club (1999) |
| heist crew pulls off an impossible robbery | 1.00 | 3.76 | Ocean's Eleven (2001), Out of Sight (1998), Usual Suspects, The (1995) |
| space opera with aliens and starships | 1.00 | 3.76 | Star Wars: Episode VI - Return of the Jedi (1983), Star Wars: Episode IV - A New Hope (1977), Star Wars: Episode V - The Empire Strikes Back (1980) |
| feel good comedy about friendship | 1.00 | 3.00 | Calendar Girls (2003), Trippin' (1999), Employee of the Month (2006) |
| haunted house horror where something is in the basement | 1.00 | 2.65 | Amityville Curse, The (1990), Ju-on: The Grudge (2002), Amityville 3-D (1983) |
| courtroom drama about a wrongful conviction | 1.00 | 3.14 | Margaret (2011), Recount (2008), Ghosts of Mississippi (1996) |
| animated film for children about talking animals | 1.00 | 2.35 | The Tale of the Bunny Picnic (1986), Zeus and Roxanne (1997), Baby-Sitters Club, The (1995) |
| war film about soldiers on the front line | 1.00 | 3.76 | Battle for Haditha (2007), Pork Chop Hill (1959), Objective, Burma! (1945) |
| romantic story where two people meet in a foreign city | 0.90 | 3.78 | Erin Brockovich (2000), Kids (1995), Schindler's List (1993) |
| documentary about music | 1.00 | 3.80 | George Harrison: Living in the Material World (2011), American Hardcore (2006), Genghis Blues (1999) |
