"""
Pulls MLB Pipeline prospect rankings (overall Top 100 + each org's Top 30)
into the prospects table, then attaches a real season stat line (GP, K%,
BB%, OPS/ERA) from the official MLB Stats API for each prospect.

There's no official rankings API, but each MLB Pipeline rankings page
(mlb.com/milb/prospects/<team-slug>) ships its full list inside the HTML: a
`data-init-state` attribute holds the JSON the page's own code renders from
(only the first 5 rows are drawn until "Show Full List" is clicked, but all
30 - or 100 - are in that JSON). So a plain HTTP GET per page is enough; no
browser is needed, which is what lets this run on the hosted machine.

Each prospect's MLBAM person id comes with the ranking entry. That id is
then used to pull real season stats from statsapi.mlb.com at the sportId
matching their Pipeline-listed level (AAA/AA/A+/A/ROK) - genuinely official
numbers, not scraped off the rankings page (which has none).

Usage:
    python pulls/daily/pull_prospects.py [SEASON]   # defaults to 2026

31 page loads (Top 100 + 30 team pages), then a ~900-prospect stat-line
lookup against statsapi on a small thread pool. Upserts, so re-running
(e.g. after a midseason Pipeline update) just refreshes everything. Run by
the scheduled refresh (api/pull.py) once a week.
"""
import html
import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

DB_PATH = os.environ.get("DB_PATH", "mlb_data.db")
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from config.model_policy import CURRENT_SEASON as DEFAULT_SEASON  # noqa: E402  (bump CURRENT_SEASON, not this)

TEAM_SLUGS = {
    "Arizona Diamondbacks": "dbacks",
    "Athletics": "athletics",
    "Atlanta Braves": "braves",
    "Baltimore Orioles": "orioles",
    "Boston Red Sox": "redsox",
    "Chicago Cubs": "cubs",
    "Chicago White Sox": "whitesox",
    "Cincinnati Reds": "reds",
    "Cleveland Guardians": "guardians",
    "Colorado Rockies": "rockies",
    "Detroit Tigers": "tigers",
    "Houston Astros": "astros",
    "Kansas City Royals": "royals",
    "Los Angeles Angels": "angels",
    "Los Angeles Dodgers": "dodgers",
    "Miami Marlins": "marlins",
    "Milwaukee Brewers": "brewers",
    "Minnesota Twins": "twins",
    "New York Mets": "mets",
    "New York Yankees": "yankees",
    "Philadelphia Phillies": "phillies",
    "Pittsburgh Pirates": "pirates",
    "San Diego Padres": "padres",
    "San Francisco Giants": "giants",
    "Seattle Mariners": "mariners",
    "St. Louis Cardinals": "cardinals",
    "Tampa Bay Rays": "rays",
    "Texas Rangers": "rangers",
    "Toronto Blue Jays": "bluejays",
    "Washington Nationals": "nationals",
}

# MLB Stats API sportId for each Pipeline-listed level. Includes MLB (1) -
# some Top 30 prospects are rookie-eligible call-ups already in the majors.
LEVEL_TO_SPORT_ID = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "ROK": 16, "MLB": 1}

RANKINGS_URL = "https://www.mlb.com/milb/prospects"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
INIT_STATE = re.compile(r'data-init-state="([^"]*)"')
RANKINGS_KEY = "getPlayerRankingsFromSelection"
TOP100_SIZE = 100
ORG_LIST_SIZE = 30


def _is_pitcher(position):
    return position.startswith("RHP") or position.startswith("LHP")


def _fetch_rankings(session, url, expected):
    """Returns one page's ranked prospects as dicts, read from the JSON embedded in the HTML.
    Raises if the page does not carry the full list, so a layout change on mlb.com fails the
    pull loudly instead of quietly storing a partial ranking."""
    last_error = None
    for _ in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            last_error = e
    else:
        raise RuntimeError(f"could not load {url}: {last_error}")

    for m in INIT_STATE.finditer(resp.text):
        payload = json.loads(html.unescape(m.group(1))).get("payload") or {}
        for key, entries in (payload.get("ROOT_QUERY") or {}).items():
            if key.startswith(RANKINGS_KEY):
                if len(entries) != expected:
                    raise RuntimeError(f"{url}: expected {expected} ranked prospects, found {len(entries)}")
                return [_prospect_row(payload, e) for e in entries]
    raise RuntimeError(f"{url}: no embedded rankings found (page layout changed?)")


def _resolve(payload, node):
    """The embedded JSON stores shared objects once and points at them with {"__ref": key}."""
    return payload[node["__ref"]] if isinstance(node, dict) and "__ref" in node else node


def _prospect_row(payload, entry):
    entity = entry["playerEntity"]
    person = _resolve(payload, entity["player"])
    roster = _resolve(payload, person.get("activeRoster"))
    sport = _resolve(payload, roster.get("sport")) if roster else None
    eta = entity.get("eta") or ""
    return {
        "mlb_id": int(person["id"]),
        "rank": int(entry["rank"]),
        "name": f"{person['useName']} {person['useLastName']}",
        "position": entity["position"],
        "level": sport["abbreviation"] if sport else "-",
        "eta": int(eta) if eta.isdigit() else None,
        "age": person.get("currentAge"),
        "height_weight": f"{person['height']} / {person['weight']} lbs",
        "bats": person.get("batSideCode"),
        "throws": person.get("pitchHandCode"),
    }


def scrape_rankings(season):
    """Returns (rows, top100_ranks) - rows is a list of prospect dicts
    (one per org Top 30 appearance), top100_ranks maps mlb_id -> overall rank."""
    session = requests.Session()

    print("Reading overall Top 100...", end=" ", flush=True)
    top100_ranks = {r["mlb_id"]: r["rank"] for r in _fetch_rankings(session, RANKINGS_URL, TOP100_SIZE)}
    print(f"{len(top100_ranks)} ranked players")

    rows = []
    for team_name, slug in TEAM_SLUGS.items():
        print(f"Reading {team_name} Top 30...", end=" ", flush=True)
        team_rows = _fetch_rankings(session, f"{RANKINGS_URL}/{slug}", ORG_LIST_SIZE)
        for r in team_rows:
            r["team_name"] = team_name
            r["org_rank"] = r.pop("rank")
        rows += team_rows
        print(f"{len(team_rows)} rows")

    return rows, top100_ranks


def fetch_stat_line(session, mlb_id, level, is_pitcher, season):
    """Returns (gp, k_pct, bb_pct, ops, era) from statsapi at the sportId
    matching the prospect's Pipeline-listed level, or all-None if the level
    isn't recognized or the player has no season stats there (e.g. injured
    all year, just signed)."""
    sport_id = LEVEL_TO_SPORT_ID.get(level)
    if sport_id is None:
        return None, None, None, None, None

    group = "pitching" if is_pitcher else "hitting"
    url = f"{STATSAPI_BASE}/people/{mlb_id}/stats"
    params = {"stats": "season", "group": group, "season": season, "sportId": sport_id}
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None, None, None, None, None

    splits = (data.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return None, None, None, None, None
    stat = splits[0]["stat"]

    gp = stat.get("gamesPlayed")
    if is_pitcher:
        bf = stat.get("battersFaced") or 0
        k_pct = round(100 * stat["strikeOuts"] / bf, 1) if bf else None
        bb_pct = round(100 * stat["baseOnBalls"] / bf, 1) if bf else None
        era = float(stat["era"]) if stat.get("era") not in (None, "-.--") else None
        return gp, k_pct, bb_pct, None, era
    else:
        pa = stat.get("plateAppearances") or 0
        k_pct = round(100 * stat["strikeOuts"] / pa, 1) if pa else None
        bb_pct = round(100 * stat["baseOnBalls"] / pa, 1) if pa else None
        ops = float(stat["ops"]) if stat.get("ops") not in (None, "-.--") else None
        return gp, k_pct, bb_pct, ops, None


def run(season: int):
    conn = sqlite3.connect(DB_PATH)
    team_ids = {name: tid for tid, name in conn.execute("SELECT team_id, name FROM teams")}

    rows, top100_ranks = scrape_rankings(season)

    # Some prospects have offseason/complex-league stints on top of their
    # main Pipeline-listed level, or appear on the team page more than once
    # across a re-scrape; keep the first (highest org-ranked) occurrence.
    by_id = {}
    for r in rows:
        by_id.setdefault(r["mlb_id"], r)

    print(f"\nFetching season stats for {len(by_id)} prospects (10 concurrent)...")
    session = requests.Session()  # thread-safe for concurrent .get() - urllib3's
                                   # connection pooling underneath is what makes this
                                   # a real speedup rather than just more overhead
    now = datetime.now(timezone.utc).isoformat()
    unmatched_teams = set()
    unmapped_levels = set()

    def fetch_one(mlb_id, r):
        is_pitcher = _is_pitcher(r["position"])
        gp, k_pct, bb_pct, ops, era = fetch_stat_line(session, mlb_id, r["level"], is_pitcher, season)
        return mlb_id, r, is_pitcher, (gp, k_pct, bb_pct, ops, era)

    results = {}
    with ThreadPoolExecutor(10) as pool:
        futures = {pool.submit(fetch_one, mlb_id, r): mlb_id for mlb_id, r in by_id.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            mlb_id, r, is_pitcher, stat_line = fut.result()
            results[mlb_id] = (r, is_pitcher, stat_line)
            if i % 100 == 0:
                print(f"  ...{i}/{len(by_id)}")

    db_rows = []
    for mlb_id, r in by_id.items():  # stable order, not thread-completion order
        _r, is_pitcher, (gp, k_pct, bb_pct, ops, era) = results[mlb_id]

        team_id = team_ids.get(r["team_name"])
        if team_id is None:
            unmatched_teams.add(r["team_name"])
        if r["level"] not in LEVEL_TO_SPORT_ID:
            unmapped_levels.add(r["level"])

        headshot = f"https://img.mlbstatic.com/mlb-photos/image/upload/c_fill,g_auto/w_360/v1/people/{mlb_id}/headshot/milb/current"

        db_rows.append((
            mlb_id, season, r["name"], r["position"], int(is_pitcher), team_id,
            r["org_rank"], top100_ranks.get(mlb_id), r["level"], r["eta"], r["age"],
            r["height_weight"], r["bats"], r["throws"], headshot,
            season, gp, k_pct, bb_pct, ops, era, now,
        ))

    conn.executemany(
        "INSERT INTO prospects (mlb_id, season, name, position, is_pitcher, team_id, "
        "org_rank, top100_rank, level, eta, age, height_weight, bats, throws, headshot_url, "
        "stats_season, gp, k_pct, bb_pct, ops, era, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(mlb_id, season) DO UPDATE SET "
        "name=excluded.name, position=excluded.position, is_pitcher=excluded.is_pitcher, "
        "team_id=excluded.team_id, org_rank=excluded.org_rank, top100_rank=excluded.top100_rank, "
        "level=excluded.level, eta=excluded.eta, age=excluded.age, "
        "height_weight=excluded.height_weight, bats=excluded.bats, throws=excluded.throws, "
        "headshot_url=excluded.headshot_url, stats_season=excluded.stats_season, gp=excluded.gp, "
        "k_pct=excluded.k_pct, bb_pct=excluded.bb_pct, ops=excluded.ops, era=excluded.era, "
        "updated_at=excluded.updated_at",
        db_rows,
    )
    # Anyone not on a list anymore (graduated, traded off, dropped) still has a row from an earlier
    # run, with a rank that now belongs to someone else. Every list was read in full above, so
    # whatever this run did not touch is no longer ranked.
    dropped = conn.execute("DELETE FROM prospects WHERE season = ? AND updated_at < ?", (season, now)).rowcount
    conn.commit()
    conn.close()

    have_stats = sum(1 for row in db_rows if row[16] is not None)
    print(f"\nUpserted {len(db_rows)} prospects ({len(top100_ranks)} on the Top 100, "
          f"{have_stats} with a season stat line).")
    print(f"Removed {dropped} prospect(s) no longer on any list.")
    if unmatched_teams:
        print(f"  WARNING: {len(unmatched_teams)} team name(s) didn't match the teams table: {sorted(unmatched_teams)}")
    if unmapped_levels:
        print(f"  WARNING: {len(unmapped_levels)} level string(s) had no sportId mapping (stats skipped): {sorted(unmapped_levels)}")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
