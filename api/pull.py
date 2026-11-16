"""
Data refresh pipeline - rebuilt from scratch (previous version's button
reportedly still didn't work reliably for the user, and was slow on top of
that). Two ways this gets triggered, both funnel through the exact same
`_run()`:

- Automatically, at server startup and every day at 6 AM (api/scheduler.py)
  - the primary mechanism now. The user shouldn't need to click anything
  for the site to stay current.
- The nav's "Refresh data" button (POST /api/admin/pull-data) - a manual
  "check right now" for when someone doesn't want to wait for the schedule.

There is deliberately no separate "fast" vs "full" pull mode anymore. The
previous version tried to make the button fast by only checking a recent
date window, which is exactly the kind of shortcut that can silently miss
a real gap. Instead:
  - pull_statcast.py's own `ingested_dates` skip-list means checking the
    ENTIRE season history on every single call costs almost nothing when
    everything's already covered (a few thousand cheap set lookups) - so
    every call here checks the full history, not a shortcut window.
  - pull_players_games.py's get_distinct_player_ids/get_distinct_games are
    gap-aware (recent activity UNION anything ever missing entirely), so
    a normal run stays fast (network calls only for what's actually new or
    genuinely never-seen) without needing a separate "full rescan" mode
    to catch real gaps.
The only thing that still varies by trigger is how thoroughly the display
cache gets rewarmed afterward (`full_warmup`) - see warmup.py's docstring -
since that's a pure UI-responsiveness tradeoff, not a data-correctness one.
"""
import logging
import os
import sqlite3
import json
import subprocess
import sys
import threading
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("refresh")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulls.daily import pull_players_games
from pulls.daily import pull_team_stats
from pulls.daily import pull_remaining_schedule
from analytics import boxscore

from api import cache
from api.db import DB_PATH
from api.warmup import warm_cache_sync
from config.model_policy import CURRENT_SEASON
from config.season_phase import skip_team_stat_pulls as _skip_team_stats, is_postseason as _is_postseason

# A few days before 2021's actual opening day (~April 1) - close enough
# that this doesn't sweep in a needless extra month of guaranteed-empty
# offseason date checks, but early enough to safely cover every backfilled
# season. Checking already-covered dates is cheap (see get_uningested_dates
# in pull_statcast.py) but a date that's NEVER been checked before still
# costs one real (if fast, 0-row) pybaseball call before it gets marked and
# becomes cheap forever after - so this constant is a genuine, if small,
# one-time cost lever, not just documentation.
EARLIEST_SEASON_START = date(2021, 3, 20)
# Mirrors pull_statcast.RECENT_WINDOW_DAYS - not imported from there directly
# because pull_statcast pulls in pybaseball (and transitively matplotlib/Pillow),
# which would otherwise be loaded at every boot just for this one module-level
# constant. pull_statcast itself is imported lazily in _run() below.
PLAYERS_GAMES_RECENT_WINDOW_DAYS = 3

# POSTSEASON_MODE is now derived from config/season_phase.py (SEASON_PHASE env var
# or the default set there). Do not change this line; change SEASON_PHASE instead.
POSTSEASON_MODE = _is_postseason()

_status = {"state": "idle", "detail": "", "started_at": None, "finished_at": None}
_lock = threading.Lock()

# Persisted (survives a restart, unlike _status above) so a crash-loop -
# the process gets OOM-killed mid-warmup, restarts, and would otherwise
# immediately re-trigger the exact same pull+clear+rebuild every time -
# can be detected and the startup pull skipped. Lives next to the DB (same
# persistent volume), not in the DB itself, so it doesn't need a
# schema migration or risk corrupting real data.
_PULL_STATE_FILE = Path(DB_PATH).resolve().parent / "pull_state.json"
RESTART_PULL_SKIP_WINDOW_SECONDS = 2 * 3600  # 2 hours


def last_pull_finished_at():
    """Unix timestamp of the last pull cycle that completed successfully
    (didn't except), or None if unknown (first boot ever, or the state
    file is missing/corrupt - always safe to treat as None, which just
    means "pull as normal")."""
    try:
        return json.loads(_PULL_STATE_FILE.read_text())["finished_at"]
    except Exception:
        return None


def _record_pull_finished():
    try:
        _PULL_STATE_FILE.write_text(json.dumps({"finished_at": time.time()}))
    except Exception as e:
        log.warning("could not persist pull_state.json (non-fatal): %s", e)


def get_status():
    return dict(_status)


def _precompute_tunneling_pairs(season):
    """Regenerate the season's tunneling parquets so the player page reads
    fresh precomputed data. Runs in a subprocess (its memory is released when
    it exits) and has its own failure boundary: an error is logged and never
    blocks warm-up."""
    script = Path(__file__).resolve().parent.parent / "tools" / "precompute_tunneling_pairs.py"
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(script), str(season)],
            capture_output=True, text=True, timeout=1200,
            env={**os.environ, "DB_PATH": str(DB_PATH)},
        )
        elapsed = time.monotonic() - started
        if result.returncode == 0:
            summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            log.info("tunneling precompute ok: season=%s elapsed=%.1fs %s", season, elapsed, summary)
        else:
            log.error("tunneling precompute failed: season=%s exit=%s elapsed=%.1fs %s",
                      season, result.returncode, elapsed, result.stderr[-1000:])
    except Exception as e:
        log.error("tunneling precompute failed: season=%s elapsed=%.1fs %s: %s",
                  season, time.monotonic() - started, type(e).__name__, e)


# Caches of pages built from tables the daily pull refreshes with or without new pitches.
DATA_TABLE_CACHE_MODULES = (
    "api.routers.team_card", "api.routers.probable_pitchers", "api.routers.playoffs",
)


PROSPECT_REFRESH_DAYS = 7


def _prospects_due(season) -> bool:
    """Prospects refresh weekly, on Mondays. They are also due whenever the stored rankings are more
    than a week old, so a Monday run that failed or never happened (machine down, deploy) is made up
    at the next scheduled refresh instead of waiting another week."""
    if date.today().weekday() == 0:
        return True
    conn = sqlite3.connect(DB_PATH)
    try:
        last = conn.execute("SELECT MAX(updated_at) FROM prospects WHERE season = ?", (season,)).fetchone()[0]
    finally:
        conn.close()
    if last is None:
        return True
    return datetime.now(timezone.utc) - datetime.fromisoformat(last) > timedelta(days=PROSPECT_REFRESH_DAYS)


PROSPECT_SCRIPT_TIMEOUT_SECONDS = 15 * 60


def _refresh_prospects(season):
    """Rankings, then scouting reports. Each runs as its own process with a time limit, so a hang or
    crash cannot stall or take down the site, and its memory is returned when it exits. Each script
    writes its table in one transaction at the very end, so a run that fails or is stopped leaves the
    data already stored untouched. Scouting still runs when rankings failed (it works from the
    prospects already stored). Raises after both have had their turn if either failed."""
    scripts_dir = Path(__file__).resolve().parent.parent / "pulls" / "daily"
    errors = []
    for script, args in (("pull_prospects.py", [str(season)]), ("pull_prospect_scouting.py", [])):
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, str(scripts_dir / script), *args],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=PROSPECT_SCRIPT_TIMEOUT_SECONDS, cwd=str(scripts_dir.parent.parent),
                env={**os.environ, "DB_PATH": str(DB_PATH), "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{script} stopped after {PROSPECT_SCRIPT_TIMEOUT_SECONDS}s")
            continue
        elapsed = time.monotonic() - started
        if result.returncode == 0:
            log.info("%s ok in %.0fs: %s", script, elapsed, " | ".join(result.stdout.strip().splitlines()[-4:]))
        else:
            errors.append(f"{script} exit {result.returncode} after {elapsed:.0f}s: {result.stderr.strip()[-1500:]}")
    # Whatever was stored may have changed: drop the cached prospect pages (rebuilt by the warm-up).
    cache.clear_modules(("api.routers.prospects",))
    if errors:
        raise RuntimeError("; ".join(errors))


def _step(label, fn, failed):
    """Runs one refresh step. A step that raises is logged and recorded, and the rest still run: each
    step refreshes its own tables from its own source, so one source being down costs only that
    step, which keeps the data it already had. Returns the step's result, or None if it failed."""
    log.info("%s...", label)
    _status["detail"] = f"{label[0].upper()}{label[1:]}..."
    try:
        return fn()
    except Exception:
        log.exception("refresh step failed (the rest continue): %s", label)
        failed.append(label)
        return None


def _pull_steps(full_warmup, season):
    """Runs every refresh step and returns (pull_succeeded, new_pitches, summary)."""
    end = date.today()
    failed = []

    def pitches():
        # Checks the full history every time (see module docstring); the ingested_dates skip-list
        # makes that cheap. Imported here because it pulls in pybaseball (see the note above).
        from pulls.daily import pull_statcast
        return pull_statcast.run(EARLIEST_SEASON_START.isoformat(), end.isoformat())

    def players_games():
        recent_cutoff = (end - timedelta(days=PLAYERS_GAMES_RECENT_WINDOW_DAYS)).isoformat()
        return pull_players_games.run(since_date=recent_cutoff)

    def upcoming_schedule():
        from data.mlb_stats_api import schedule as _mlb_schedule
        _mlb_schedule(end.isoformat(), (end + timedelta(days=14)).isoformat(), db_path=str(DB_PATH))

    def rosters():
        from pulls.daily.refresh_mlb_stats import run as _refresh_mlb_stats
        _refresh_mlb_stats(season=season, skip_news=False, postseason=POSTSEASON_MODE)

    def prospects():
        # Skipped on manual refresh clicks; weekly during scheduled runs.
        if full_warmup and _prospects_due(season):
            _refresh_prospects(season)

    pitches_result = _step("checking for new pitch data", pitches, failed)
    # The pitch check is what makes this a successful pull: it decides whether the caches are rebuilt
    # and whether the next boot may skip pulling. The other steps only refresh their own tables.
    pull_succeeded = pitches_result is not None
    new_pitches = pitches_result["rows_inserted"] if pull_succeeded else 0
    games_result = (_step("updating players/games reference data", players_games, failed)
                    or {"games_updated": 0, "players_updated": 0})
    if not POSTSEASON_MODE:
        _step("updating team stats", lambda: pull_team_stats.run(season), failed)
    else:
        log.info("postseason mode - skipping team stats (RS records frozen)")
    _step("updating upcoming schedule / probable pitchers", upcoming_schedule, failed)
    _step("updating remaining schedule (SOS/playoffs)", lambda: pull_remaining_schedule.run(season), failed)
    _step("updating rosters + schedule via MLB Stats API", rosters, failed)
    _step("updating prospect rankings + scouting", prospects, failed)

    if pull_succeeded:
        _record_pull_finished()
    # These pages read tables the steps above refresh (schedule, rosters, team stats) and do not depend
    # on new pitches, so the "only clear when pitches arrived" rule below would leave them stale. They
    # are cheap to rebuild.
    cache.clear_modules(DATA_TABLE_CACHE_MODULES)
    if not pull_succeeded:
        summary = "Pitch data check failed."
    elif new_pitches == 0:
        summary = f"No new pitches found - already up to date through {end.isoformat()}."
    else:
        summary = (
            f"Up to date through {end.isoformat()}: {new_pitches:,} new pitches, "
            f"{games_result['games_updated']} game(s) updated, "
            f"{games_result['players_updated']} player(s) checked."
        )
    if failed:
        summary += f" Failed and kept their previous data: {', '.join(failed)}."
    log.info("pull done - %s", summary)
    return pull_succeeded, new_pitches, summary


def _run(full_warmup):
    season = CURRENT_SEASON
    try:
        pull_succeeded, new_pitches, summary = _pull_steps(full_warmup, season)
    except Exception as e:
        # The steps guard themselves; this covers the bookkeeping around them, so the warm-up below
        # and the lock release always happen.
        log.exception("pull failed outside its steps")
        pull_succeeded, new_pitches, summary = False, 0, f"Refresh failed: {e}"

    # Warm-up runs in its OWN try/except, regardless of whether the pull
    # above succeeded - a network blip from Savant/the MLB Stats API used
    # to mean warm_cache_sync() never ran at all (it was the last step
    # inside the pull's own try block), so the site served nothing warmed
    # until the NEXT scheduled cycle. Now a pull failure just means this
    # cycle doesn't learn about new data; whatever's already in the DB
    # still gets (re)warmed.
    try:
        log.info("refreshing computed stats...")
        _status["detail"] = "Refreshing computed stats..."
        # Every full refresh whose pull succeeded (the daily 4 AM run, and a boot that pulls) empties
        # the CURRENT season's caches - with or without new pitches, in every season phase - and the
        # warm-up below rebuilds them, so the player pages are rebuilt from the data every single day.
        # Finished seasons are untouched (unchanged, and slow to rebuild on the small machine), as are
        # the percentile pools persisted on disk. A failed pull leaves the existing caches alone.
        if full_warmup and pull_succeeded:
            from analytics import stuff_model

            cache.clear_current_season(CURRENT_SEASON)
            stuff_model.clear_cache(CURRENT_SEASON)
            boxscore.clear_cache()
        elif not full_warmup and new_pitches > 0:
            # Manual refresh button, and something actually changed: only
            # clear the daily-data caches. Model scoring caches (Stuff+,
            # Tunneling+, Deception+, Location+) don't meaningfully change
            # because of one day's new pitches — clearing and recomputing
            # them cold costs ~25-35s of LightGBM scoring that the user's
            # already paid for today. Keep them warm. Only clear what
            # actually needs to reflect new data: box scores (completed
            # games), leaderboards (daily top performers), standings.
            from api.routers.top_performers import _leaderboard
            from api.routers.standings import _standings
            boxscore.clear_cache()
            _leaderboard.clear()
            _standings.clear()
        # else: manual refresh with nothing new, or the pull failed - skip clearing, just warm what is cold.
        # Only when this pull injected new pitches: the precompute loads a
        # full season of pitches, which is memory-heavy, and the readers use
        # the committed parquet whenever it exists.
        if pull_succeeded and new_pitches > 0:
            _precompute_tunneling_pairs(season)
        warm_cache_sync(full=full_warmup)
        _status["state"] = "done" if pull_succeeded else "error"
        _status["detail"] = summary if pull_succeeded else summary + " (warmed from existing data)"
        log.info("warm-up done")
    except Exception as e:
        _status["state"] = "error"
        _status["detail"] = f"Warm-up failed: {e}"
        log.error("warm_cache_sync failed (%s): %s", type(e).__name__, e)
        traceback.print_exc()
    finally:
        _status["finished_at"] = date.today().isoformat()
        _lock.release()


def start_pull(full_warmup=False):
    """Returns True if a pull was started, False if one was already
    running. `full_warmup=True` (used by the scheduled/startup triggers,
    where nobody's waiting) also rewarms the L/R percentile splits, not
    just the default ALL view - see warmup.py."""
    if not _lock.acquire(blocking=False):
        return False
    _status["state"] = "running"
    _status["detail"] = "Starting..."
    _status["started_at"] = date.today().isoformat()
    _status["finished_at"] = None
    threading.Thread(target=_run, args=(full_warmup,), daemon=True).start()
    return True
