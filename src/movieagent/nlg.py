"""Turning tool evidence into English.

Every sentence below is assembled from values a tool returned. There is no
free-form generation and no outside movie knowledge, which is the point: if the
assistant says "12 of your closest neighbours rated it 4.6", that number came
out of `ratings.csv` and can be checked.
"""

from __future__ import annotations


def _stars(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _movie_line(rec: dict) -> str:
    year = f" ({rec['year']})" if rec.get("year") else ""
    return f"{rec['title']}{year} - {rec.get('genres', '')}"


def _audience(rec: dict) -> str:
    n = rec.get("n_ratings") or 0
    if not n:
        return "no ratings in this dataset"
    tail = " (thin evidence)" if n < 10 else ""
    return f"audience {_stars(rec.get('mean_rating'))} from {n} ratings{tail}"


def profile(p: dict) -> str:
    if "error" in p:
        return p["error"]
    gen = p["generosity_vs_all_users"]
    mood = ("a generous rater" if gen > 0.25 else
            "a harsh rater" if gen < -0.25 else "a fairly average rater")
    favs = ", ".join(f"{g['genre']} ({g['avg']:.2f} over {g['n']})" for g in p["favourite_genres"]) or "no clear standout"
    watched = ", ".join(f"{g['genre']} x{g['n']}" for g in p["most_watched_genres"])
    top = "; ".join(f"{t['title']} {t['rating']:.1f}" for t in p["top_rated"][:5])
    era = p["era"]
    era_txt = ""
    if era["median_year"]:
        era_txt = (f" Median release year in your history is {era['median_year']}"
                   f" ({int(era['share_pre_1990']*100)}% pre-1990).")
    return (
        f"You're user {p['userId']}: {p['n_ratings']} ratings, average {p['avg_rating']:.2f} "
        f"({mood}, {gen:+.2f} vs the dataset mean), spread {p['rating_spread_std']:.2f}.\n"
        f"Most watched: {watched}.\n"
        f"Rated highest relative to your own baseline: {favs}.\n"
        f"Your top films: {top}.{era_txt}"
    )


def recommendations(payload: dict, reasons: dict[int, dict], profile_data: dict | None) -> str:
    recs = payload.get("recommendations") or []
    if not recs:
        return ("Nothing survived those filters. Try loosening the genre constraint or "
                "lowering the minimum rating count.")

    head = []
    if profile_data and "n_ratings" in profile_data:
        head.append(f"Working from your {profile_data['n_ratings']} ratings "
                    f"(you average {profile_data['avg_rating']:.2f})")
    if payload.get("seeded_on"):
        head.append(f"steered toward {', '.join(payload['seeded_on'])}")
    cons = payload.get("constraints") or {}
    if cons.get("exclude_genres"):
        head.append(f"excluding {', '.join(cons['exclude_genres'])}")
    if cons.get("include_genres"):
        head.append(f"restricted to {', '.join(cons['include_genres'])}")
    lines = [", ".join(head) + ":" if head else "Here's what I'd put on:"]

    for i, rec in enumerate(recs, 1):
        lines.append(f"\n{i}. {_movie_line(rec)}")
        cap = " (capped)" if rec.get("predicted_rating_capped") else ""
        lines.append(f"   predicted {rec['predicted_rating']:.1f}/5{cap} for you - {_audience(rec)}")
        why = reasons.get(rec["movieId"])
        if why:
            lines.append("   " + _why_sentence(why))
        else:
            ev = rec.get("evidence", {})
            if ev.get("neighbour_support", 0) > 0:
                lines.append(f"   backed by similar-user signal (support {ev['neighbour_support']:.1f})")
    if any(r.get("predicted_rating_capped") for r in recs):
        lines.append("\n(\"capped\" = the model predicted above 5.0 and I clipped it. You rate "
                     "generously, so predicted scores bunch at the ceiling; use the ordering and "
                     "the evidence rather than the absolute number.)")
    if payload.get("unresolved_titles"):
        lines.append(f"\n(I couldn't find these titles in the catalogue: "
                     f"{', '.join(payload['unresolved_titles'])}.)")
    return "\n".join(lines)


def _why_sentence(why: dict) -> str:
    bits = []
    peers = why.get("peers") or {}
    if peers.get("n_similar_users_who_rated_it"):
        delta = peers.get("similar_user_mean_vs_own_average")
        d = f", {delta:+.2f} vs their own baseline" if delta is not None else ""
        bits.append(f"{peers['n_similar_users_who_rated_it']} of your closest neighbours "
                    f"rated it {peers['similar_user_mean']:.2f}{d}")
    drivers = why.get("because_you_rated") or []
    if drivers:
        d = drivers[0]
        bits.append(f"people who liked it also liked {d['title']}, which you gave {d['your_rating']:.1f}")
    content = why.get("plot_resembles_your_favourites") or []
    if content and len(bits) < 2:
        c = content[0]
        bits.append(f"its plot reads like {c['title']} ({c['your_rating']:.1f} from you)")
    fit = [g for g in (why.get("genre_fit") or []) if g.get("vs_your_average") is not None]
    if fit:
        best = max(fit, key=lambda g: g["vs_your_average"])
        if best["vs_your_average"] > 0.05:
            bits.append(f"{best['genre']} runs {best['vs_your_average']:+.2f} above your average")
    return "why: " + "; ".join(bits[:3]) if bits else "why: thin evidence - mostly a popularity fallback"


def explanation(why: dict) -> str:
    if "error" in why:
        return why["error"] + (f" Did you mean: {', '.join(why['did_you_mean'])}?" if why.get("did_you_mean") else "")
    m = why["movie"]
    cap = (f" (raw model output {why['predicted_rating_raw']:.2f}, clipped to the 5.0 ceiling)"
           if why.get("predicted_rating_raw", 0) > 5.02 else "")
    out = [f"{_movie_line(m)} - predicted {why['predicted_rating']:.2f}/5{cap} for you, {_audience(m)}.", ""]

    drivers = why.get("because_you_rated") or []
    if drivers:
        out.append("From your own ratings (people who rated these the way you did tend to rate this film similarly):")
        for d in drivers:
            out.append(f"  - {d['title']} - you gave {d['your_rating']:.1f} (taste overlap {d['taste_overlap']:.2f})")
    else:
        out.append("Nothing in your history has a strong audience-overlap signal with this film.")

    content = why.get("plot_resembles_your_favourites") or []
    if content:
        out.append("")
        out.append("Films in your history with the most similar plots:")
        for c in content[:3]:
            out.append(f"  - {c['title']} - you gave {c['your_rating']:.1f} (plot similarity {c['plot_similarity']:.2f})")

    peers = why.get("peers") or {}
    if peers.get("n_similar_users_who_rated_it"):
        out.append("")
        out.append(f"Users with taste closest to yours: {peers['n_similar_users_who_rated_it']} of them rated it, "
                   f"average {peers['similar_user_mean']:.2f} "
                   f"({peers['similar_user_mean_vs_own_average']:+.2f} against their own averages). "
                   f"Everyone else: {peers['everyone_mean']:.2f} from {peers['everyone_n']} ratings.")
    else:
        out.append("")
        out.append("Caveat: none of your nearest neighbours has rated it, so the prediction leans on "
                   "content similarity and overall popularity rather than on people like you.")

    fit = why.get("genre_fit") or []
    if fit:
        txt = ", ".join(f"{g['genre']} {g['vs_your_average']:+.2f} over {g['n']} films"
                        for g in fit if g.get("vs_your_average") is not None)
        if txt:
            out.append("")
            out.append(f"Genre fit versus your own average: {txt}.")
    if why.get("plot_keywords"):
        out.append(f"Plot keywords: {', '.join(why['plot_keywords'][:6])}.")
    return "\n".join(out)


def peer_opinion(p: dict) -> str:
    if "error" in p:
        return p["error"] + (f" Did you mean: {', '.join(p['did_you_mean'])}?" if p.get("did_you_mean") else "")
    m = p["movie"]
    if not p["n_similar_users_who_rated_it"]:
        return (f"None of your nearest neighbours has rated {m['title']}. "
                f"Across everyone in the dataset it sits at {_stars(p['everyone_mean'])} "
                f"from {p['everyone_n']} ratings, but that's the crowd, not your crowd.")
    gap = p["similar_user_mean"] - (p["everyone_mean"] or 0)
    verdict = ("rates it noticeably higher than the crowd at large" if gap > 0.2 else
               "rates it about the same as the crowd at large" if abs(gap) <= 0.2 else
               "rates it lower than the crowd at large")
    lines = [
        f"{_movie_line(m)}",
        f"{p['n_similar_users_who_rated_it']} of the users closest to your taste have rated it: "
        f"average {p['similar_user_mean']:.2f}, which is {p['similar_user_mean_vs_own_average']:+.2f} "
        f"against their own baselines.",
        f"The whole dataset gives it {_stars(p['everyone_mean'])} from {p['everyone_n']} ratings, "
        f"so your crowd {verdict} ({gap:+.2f}).",
    ]
    if p.get("you_already_rated_it") is not None:
        lines.append(f"You've already rated it {p['you_already_rated_it']:.1f}.")
    lines.append("")
    lines.append("Individual neighbours:")
    for r in p["raters"][:6]:
        lines.append(f"  - user {r['userId']} (similarity {r['similarity']:.3f}, {r['co_rated']} films in common) "
                     f"gave it {r['rating']:.1f} ({r['vs_their_average']:+.2f} vs their average)")
    return "\n".join(lines)


def search_results(payload: dict) -> str:
    res = payload.get("results") or []
    if not res:
        return payload.get("note", "Nothing matched that.")
    f = payload.get("filters") or {}
    bits = [f"Searching plots, genres and tags for {payload['query']!r}"]
    if f.get("include_genres"):
        bits.append(f"limited to {', '.join(f['include_genres'])}")
    if f.get("exclude_genres"):
        bits.append(f"excluding {', '.join(f['exclude_genres'])}")
    if f.get("year_from") or f.get("year_to"):
        bits.append(f"years {f.get('year_from') or '...'}-{f.get('year_to') or '...'}")
    lines = [", ".join(bits) + ":"]
    for i, r in enumerate(res, 1):
        lines.append(f"\n{i}. {_movie_line(r)}")
        pred = r.get("predicted_rating_for_user")
        pred_txt = f"predicted {pred:.1f}/5 for you - " if pred is not None else ""
        lines.append(f"   {pred_txt}{_audience(r)} - plot match {r['plot_match']:.2f}")
        if r.get("why_it_matched"):
            lines.append(f"   matched on: {', '.join(r['why_it_matched'][:5])}")
    return "\n".join(lines)


def blind_spots(b: dict) -> str:
    if "error" in b:
        return b["error"]
    lines = [f"Across your {b['n_ratings']} ratings, you lean hard on "
             + ", ".join(f"{g['genre']} ({g['n']} films)" for g in b["dominant_genres"]) + "."]
    if b["unexplored_genres"]:
        lines.append("\nBarely touched:")
        for g in b["unexplored_genres"][:4]:
            entry = g.get("suggested_entry_points") or []
            picks = "; ".join(f"{e['title']} ({_stars(e['mean_rating'])} from {e['n_ratings']})" for e in entry)
            lines.append(f"  - {g['genre']}: {g['you_have_rated']} rated, "
                         f"{g['well_known_titles_available']} well-known titles available here"
                         + (f"\n      start with: {picks}" if picks else ""))
    if b["under_explored_genres"]:
        lines.append("\nLightly sampled (and how you felt about them):")
        for g in b["under_explored_genres"][:4]:
            avg = f"you average {g['your_avg_here']:.2f} there ({g['vs_your_average']:+.2f} vs your overall)" \
                if g["your_avg_here"] is not None else "no clear signal yet"
            entry = g.get("suggested_entry_points") or []
            picks = "; ".join(e["title"] for e in entry)
            lines.append(f"  - {g['genre']}: {g['you_have_rated']} rated, {avg}"
                         + (f"\n      try: {picks}" if picks else ""))
    lines.append("\nCaveat: a gap is only a blind spot if you'd enjoy the genre. "
                 "Where you have sampled a genre and rated it below your average, I read that as a preference, not a gap.")
    return "\n".join(lines)


def similar_movies(s: dict) -> str:
    if "error" in s:
        return s["error"] + (f" Did you mean: {', '.join(s['did_you_mean'])}?" if s.get("did_you_mean") else "")
    lines = [f"Neighbours of {s['movie']}:"]
    if s.get("similar_by_audience"):
        lines.append("\nBy audience overlap (people who liked one liked the other):")
        for r in s["similar_by_audience"][:5]:
            lines.append(f"  - {_movie_line(r)} - overlap {r['audience_similarity']:.2f}, {_audience(r)}")
    if s.get("similar_by_plot"):
        lines.append("\nBy plot text:")
        for r in s["similar_by_plot"][:5]:
            lines.append(f"  - {_movie_line(r)} - plot similarity {r['plot_similarity']:.2f}, {_audience(r)}")
    return "\n".join(lines)


def movie_details(m: dict, peers: dict | None = None) -> str:
    if "error" in m:
        return m["error"] + (f" Did you mean: {', '.join(m['did_you_mean'])}?" if m.get("did_you_mean") else "")
    lines = [f"{_movie_line(m)}", f"{_audience(m)} - confidence: {m['rating_confidence']}"]
    if m.get("tags"):
        lines.append(f"Viewer tags: {m['tags']}")
    lines.append("")
    lines.append(m["plot"].strip() + ("..." if len(m["plot"]) >= 590 else ""))
    if peers and peers.get("n_similar_users_who_rated_it"):
        lines.append("")
        lines.append(f"For you specifically: {peers['n_similar_users_who_rated_it']} of your nearest "
                     f"neighbours rated it {peers['similar_user_mean']:.2f} "
                     f"({peers['similar_user_mean_vs_own_average']:+.2f} vs their baselines).")
    return "\n".join(lines)


def similar_users(s: dict) -> str:
    if "error" in s:
        return s["error"]
    lines = [f"Users whose rating patterns line up with user {s['userId']}:"]
    for u in s["similar_users"]:
        favs = ", ".join(u["their_favourites"])
        lines.append(f"  - user {u['userId']}: similarity {u['similarity']:.3f} over {u['co_rated']} shared films, "
                     f"{u['n_ratings']} ratings, averages {u['avg_rating']:.2f}\n      favourites: {favs}")
    lines.append(f"\n({s['note']}.)")
    return "\n".join(lines)
