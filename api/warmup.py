"""
Pre-warms the TTL-cached league-wide pool computations (percentile pools,
Stuff+, pitcher roles, league-average rows, standings) so the player page -
and the first request after a cold cache in general - doesn't have to eat a
~15s+ full-table scan live in the request path. Runs in a background thread
so the server starts accepting requests immediately; requests that land
before warmup finishes still work, they just recompute (and cache) on
demand like before.

Score-cache parquets
--------------------
Stuff+/Tunneling+/Location+ raw-prediction parquets (written by
scripts/rescore_models.py, refreshed weekly by the scheduler) are injected
into each model's module-level raw-predictions cache at warmup time.  This
means the @ttl_cache wrappers (_stuff_plus_table, _v3_raw_predictions_table,
_location_raw_predictions_table) run their normal aggregation logic but find
the expensive LightGBM step already done — cutting cold-start model scoring
from ~90s to effectively zero.  If a parquet is missing (first deploy, or
between a model-version bump and the next weekly rescore) the functions fall
back to live scoring exactly as before.
"""
import ctypes
import gc
import logging
import os
import sqlite3
import sys
import threading
import time as _time
from pathlib import Path

log = logging.getLogger("warmup")

_INSTRUMENT = os.environ.get("BOOT_INSTRUMENT") == "1"
_step_start = None


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def _step(label):
    """Flushed start/end markers for boot-condition instrumentation.
    No-op unless BOOT_INSTRUMENT=1 (see diagnostics/boot_test for the harness
    that reads these); never runs in normal boots."""
    global _step_start
    if not _INSTRUMENT:
        return
    now = _time.monotonic()
    if label.startswith("end:") and _step_start is not None:
        elapsed = now - _step_start
        print(f"BOOTSTEP {label} elapsed={elapsed:.1f}s rss={_rss_mb():.1f}MB", flush=True, file=sys.stderr)
    else:
        _step_start = now
        print(f"BOOTSTEP {label} rss={_rss_mb():.1f}MB", flush=True, file=sys.stderr)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None  # not glibc/Linux (e.g. local Windows dev) - nothing to trim


def _trim_memory():
    """Return freed heap pages to the OS after a pandas-heavy step.

    gc.collect() alone isn't enough here: pulling a full season's pitches
    into a DataFrame and discarding it frees the memory to glibc's malloc,
    but glibc keeps those pages mapped in the process (arena fragmentation)
    rather than handing them back to the kernel - so RSS keeps climbing
    across successive warm-up steps even though each one's own data is
    short-lived. Confirmed via Docker memory tests this session: three
    back-to-back calls that each build and discard a ~700K-row DataFrame
    pushed RSS from ~290MB to 440MB+ with nothing live to show for it.
    malloc_trim(0) is glibc-specific and a no-op (never called) elsewhere.
    """
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass

from api.db import DB_PATH

from config.model_policy import CURRENT_SEASON  # single source of truth

SPLITS = [None, "L", "R"]
# Only the current season is pre-warmed — warming all backfilled seasons on
# every startup would multiply warmup cost. Complete seasons (< CURRENT_SEASON)
# are now served from disk-persisted percentile pools (see analytics/percentile_cache.py)
# so their first-request cost is near zero once built.

SCORES_DIR = Path(__file__).resolve().parent.parent / "parquets"
# Parquet-injected raw predictions are given a long TTL so they stay warm
# for the full day and aren't evicted before the API-layer TTL cache expires.
_PARQUET_TTL = 14 * 3600  # 14 hours — longer than api/cache.py's 12h TTL


def _inject_stuff_parquet(season: int):
    """Load Stuff+ raw predictions from parquet into stuff_model's cache so
    compute_stuff_plus/compute_stuff_plus_by_stand skip LightGBM scoring."""
    try:
        import pandas as pd
        from analytics.stuff_model import STUFF_ACTIVE_VERSION, _raw_predictions_cache
        path = SCORES_DIR / f"stuff_plus_{STUFF_ACTIVE_VERSION}_{season}.parquet"
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        _raw_predictions_cache[(STUFF_ACTIVE_VERSION, season)] = (df, _time.monotonic() + _PARQUET_TTL)
        log.info("stuff %s %s: loaded from parquet (%s rows)", STUFF_ACTIVE_VERSION, season, f"{len(df):,}")
        return True
    except Exception as e:
        log.warning("stuff parquet %s load failed: %s", season, e)
        return False


def _inject_tunneling_parquet(season: int):
    """Load Tunneling+ raw predictions from parquet into tunneling_v2_model's cache."""
    try:
        import pandas as pd
        from analytics import tunneling_v2_model
        path = SCORES_DIR / f"tunneling_plus_v3_{season}.parquet"
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        tunneling_v2_model._cache[f"raw_{season}"] = (df, _time.monotonic() + _PARQUET_TTL)
        log.info("tunneling v3 %s: loaded from parquet (%s rows)", season, f"{len(df):,}")
        return True
    except Exception as e:
        log.warning("tunneling parquet %s load failed: %s", season, e)
        return False


def _inject_location_parquet(season: int):
    """Load Location+ raw predictions from parquet into location_v3_model's cache."""
    try:
        import pandas as pd
        from analytics import location_v3_model
        path = SCORES_DIR / f"location_plus_v3_{season}.parquet"
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        location_v3_model._cache[f"raw_v3_{season}"] = (df, _time.monotonic() + _PARQUET_TTL)
        log.info("location v3 %s: loaded from parquet (%s rows)", season, f"{len(df):,}")
        return True
    except Exception as e:
        log.warning("location parquet %s load failed: %s", season, e)
        return False


def _inject_all_parquets(seasons):
    """Pre-load parquets for the given seasons before TTL-cache warmup runs.
    Returns (stuff_ok, tun_ok, loc_ok) booleans for each season in order, but
    in practice we just log and fall through — missing parquets are fine."""
    for season in seasons:
        _inject_stuff_parquet(season)
        _inject_tunneling_parquet(season)
        _inject_location_parquet(season)


def _warm_featured_players(conn, season, n_by_war=5, n_performers=4):
    """The player page is the landing page, so the players a visitor is most likely to open are warmed
    completely: the page payload for all three splits, similar players, last five games, pitch
    locations / hit zones, rolling xwOBA. They are today's top performers (the page opens on the
    first) and the leaders in WAR. Bounded (about 14 players, ~4 MB of cache each: warming 55 added 233 MB locally) so it cannot
    threaten the 1 GB machine; each player is guarded so one failure doesn't stop the rest."""
    from api.routers import heatmaps as _hm, hitzones as _hz, players as _pl, similar_players as _sp
    from api.routers.top_performers import _leaderboard

    wanted = []  # (player_id, role), most important first
    latest = conn.execute("SELECT MAX(game_date) FROM games").fetchone()[0]
    if latest:
        for e in (_leaderboard(conn, latest).get("entries") or [])[:n_performers]:
            wanted.append((int(e["player_id"]), e.get("role")))
    roles = _pl._pitcher_roles(conn, season)
    for col, role in (("war_batting", "batter"), ("war_pitching", "pitcher")):
        rows = conn.execute(
            f"SELECT player_id FROM player_advanced_stats WHERE season = ? AND {col} IS NOT NULL "
            f"ORDER BY {col} DESC LIMIT ?", (season, n_by_war)).fetchall()
        wanted += [(int(r[0]), role) for r in rows]
    seen, done = set(), 0
    for pid, role in wanted:
        if role not in ("pitcher", "batter") or (pid, role) in seen:
            continue
        seen.add((pid, role))
        if role == "pitcher" and pid not in roles.index:
            continue
        try:
            for split in ("ALL", "L", "R"):
                for _ in _pl._build_player_steps(conn, pid, role, split, season):
                    pass
            _pl._last_5_games(conn, season, pid, role)
            _sp.get_similar_players(pid, role=role, season=season, top_n=3, conn=conn)
            if role == "pitcher":
                for hand in ("L", "R"):
                    _hm._pitch_location_summary(conn, season, pid, hand)
            else:
                for split in ("ALL", "L", "R"):
                    _hz._hit_zone_summary(conn, season, pid, split)
                _pl.get_rolling_xwoba(pid, season=season, conn=conn)
        except Exception as e:
            log.warning("featured-player warm-up failed for %s/%s: %s", pid, role, e)
        done += 1
        if done % 10 == 0:
            _trim_memory()
    log.info("warmed %d featured player pages", done)


def _warm(conn, full):
    """`full=True` (server startup - a one-time cost, worth spending fully)
    warms every split. `full=False` (after a "Refresh data" click, which a
    user can trigger repeatedly in one sitting) warms only the ALL split -
    the L/R percentile pools are a genuine ~25s chunk of warmup time that
    most refreshes don't need paid again immediately; a user who actually
    toggles to L/R shortly after a refresh just eats the same one-time cold
    cost this app already accepts for older seasons, rather than every
    refresh paying for splits that may not get viewed at all this session."""
    from api.routers.heatmaps import _league_command_pool
    from api.routers.players import (
        _all_players, _all_pitcher_arm_angles, _approach_angle_pool, _peer_movement_table, _batter_pool, _league_avg_batter_row, _league_hr_fb,
        _pitcher_league_avg_row, _pitcher_pool, _pitcher_pool_unqualified, _pitcher_roles, _stuff_plus_by_stand_table, _stuff_plus_table,
    )
    from api.routers.deception_report import REPORT_SEASON as DECEPTION_REPORT_SEASON, _raw_predictions_table as _deception_raw_predictions_table
    from api.routers.tunneling_report import _v3_raw_predictions_table
    from api.routers.location_report import _raw_predictions_table as _location_raw_predictions_table
    from api.routers.similar_players import (
        _pitcher_arsenal_pool, _pitcher_hands, _similarity_pool_hitters, _similarity_pool_pitchers,
    )
    from api.routers.probable_pitchers import _stuff_dict, _tunneling_dict, _location_dict, _deception_dict
    from api.routers.standings import _standings
    from api.routers.stuff_model_report import _stuff_leaderboard_overall as _stuff_leaderboard
    from api.routers.team_card import _cached_rows_and_ranks, _cached_sos
    from api.routers.top_performers import _leaderboard

    season = CURRENT_SEASON

    # Entries that are still valid get a fresh lifetime: without this a nightly refresh skipped them,
    # they expired the next afternoon, and the first visitor after that paid to rebuild everything.
    from api import cache as _cache
    _cache.renew_all()

    if full:
        # Startup/scheduled refresh: full warm-up. Ordered so what the player page (the landing
        # page) needs comes first, and every phase is guarded: one failing step is logged and the
        # rest still run (an exception used to abort everything after it, leaving the site cold
        # until the next refresh).
        from analytics.percentile_cache import preload_complete_seasons
        from api.routers.career import (
            _league_hr_fb as _career_league_hr_fb, _season_batter_hr_counts, _season_pitcher_k_counts,
        )
        from api.routers.players import _league_avg_xwoba, _league_swing_profile
        from api.routers.prospects import _farm_rankings, _teams_with_prospects, _top100
        from api.routers.stuff_model_report import _stuff_leaderboard_overall
        from api.routers.tunneling_report import _v3_leaderboard
        from api.routers.trends import _league_averages_cached

        def phase_inputs():
            # Disk check for complete-season percentile pools (missing ones are built lazily).
            preload_complete_seasons(conn)
            # Only the CURRENT season's parquets are injected eagerly; past seasons load on demand.
            _inject_all_parquets([season])

        def phase_player_page_core():
            _stuff_plus_table(conn, season)
            _stuff_plus_by_stand_table(conn, season)
            _pitcher_roles(conn, season)
            _league_hr_fb(conn, season)
            _all_players(conn, season)
            for split in SPLITS:
                _pitcher_pool(conn, season, split)
                _batter_pool(conn, season, split)
                _league_avg_batter_row(conn, season, split)
                _pitcher_league_avg_row(conn, season, split)
                _pitcher_pool_unqualified(conn, season, split)
                _trim_memory()

        def phase_player_page_season_tables():
            for split in ("L", "R"):
                _league_command_pool(conn, season, split)
            _all_pitcher_arm_angles(conn, season)
            _peer_movement_table(conn, season)
            for hand in ("L", "R"):
                _approach_angle_pool(conn, season, hand)
            _league_swing_profile(conn, season)
            _league_avg_xwoba(conn, season)
            _similarity_pool_hitters(conn)
            _similarity_pool_pitchers(conn)
            _pitcher_hands(conn)
            _pitcher_arsenal_pool(conn)
            _trim_memory()

        def phase_landing_inputs():
            _standings(conn, season)
            _cached_rows_and_ranks(conn, season)
            _cached_sos(conn, season)
            # Last 10 game dates: the landing page opens on the latest day's top performer, and the
            # "top performer" badge on player pages needs these.
            for d in [r[0] for r in conn.execute(
                    "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 10").fetchall()]:
                try:
                    _leaderboard(conn, d)
                except Exception as e:
                    log.warning("_leaderboard(%s) failed: %s", d, e)
                _trim_memory()

        def phase_featured_players():
            _warm_featured_players(conn, season)

        def phase_models_and_leaderboards():
            _deception_raw_predictions_table(conn, DECEPTION_REPORT_SEASON)
            _v3_raw_predictions_table(conn, season)
            _location_raw_predictions_table(conn, season)
            _stuff_leaderboard(conn, season)
            for n in (10, 15):
                _stuff_leaderboard_overall(conn, season, n)
            _v3_leaderboard(conn, season)
            _stuff_dict(conn, season)
            _tunneling_dict(conn, season)
            _location_dict(conn, season)
            _deception_dict(conn)
            _trim_memory()

        def phase_other_pages():
            _career_league_hr_fb(conn, season)
            _season_pitcher_k_counts(conn, season)
            _season_batter_hr_counts(conn, season)
            for role in ("pitcher", "batter"):
                _league_averages_cached(conn, role, season)
            _top100(conn, season)
            _farm_rankings(conn, season)
            _teams_with_prospects(conn, season)

        for label, fn in (
            ("inputs", phase_inputs),
            ("player_page_core", phase_player_page_core),
            ("player_page_season_tables", phase_player_page_season_tables),
            ("landing_inputs", phase_landing_inputs),
            ("featured_players", phase_featured_players),
            ("models_and_leaderboards", phase_models_and_leaderboards),
            ("other_pages", phase_other_pages),
        ):
            _step(f"start:{label}")
            try:
                fn()
            except Exception:
                log.exception("warm-up phase %s failed (continuing with the rest)", label)
            _step(f"end:{label}")
        if _INSTRUMENT:
            mods = {m: (m in sys.modules) for m in ("lightgbm", "sklearn", "shap")}
            print(f"BOOTSTEP sys_modules {mods} rss={_rss_mb():.1f}MB", flush=True, file=sys.stderr)
        _step("warmup_complete")
    else:
        # Manual "Refresh data" click: model scoring caches weren't cleared
        # (one day of new pitches doesn't meaningfully change Stuff+/Tunneling+/
        # Deception+ scores and recomputing them cold costs ~25-35s). Only
        # warm the genuinely daily-fresh data: standings, today's leaderboard,
        # and the light non-model pools that ARE cleared (player list, roles, etc).
        _pitcher_roles(conn, season)
        _standings(conn, season)
        _all_players(conn, season)
        _cached_rows_and_ranks(conn, season)
        _cached_sos(conn, season)
        # Warm only today and yesterday — enough for the Top Performers page and
        # the "Last 5 Games" badge for the most-recently-played dates, without
        # the 10-15s cost of warming a full 10-date window.
        recent_dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 2"
        ).fetchall()]
        for d in recent_dates:
            try:
                _leaderboard(conn, d)
            except Exception as e:
                log.warning("_leaderboard(%s) failed: %s", d, e)


def _score_new_pitcher(conn, pitcher_id: int, season: int, model: str) -> float | None:
    """Score a single pitcher who didn't make it into the last weekly parquet
    (e.g. a mid-week debutant or a call-up after Sunday's rescore).

    Instead of recomputing the full-season distribution — which would require
    running LightGBM on hundreds of thousands of pitches — this function scores
    only the given pitcher's pitches and then normalizes using the stored
    norm_mean / norm_std from score_meta.json (the values computed when the
    parquet was last written).

    Returns the plus-scale score (100 = league mean, 10 pts = 1 SD), or None
    if the pitcher has no qualifying pitches or the meta is unavailable.

    model: one of "stuff", "tunneling", "location"
    """
    import json as _json
    meta_path = SCORES_DIR / "score_meta.json"
    if not meta_path.exists():
        return None

    meta = _json.loads(meta_path.read_text())
    season_str = str(season)

    if model == "stuff":
        from analytics.stuff_model import STUFF_ACTIVE_VERSION, _raw_predictions
        model_key = f"stuff_{STUFF_ACTIVE_VERSION}"
        season_meta = meta.get(model_key, {}).get(season_str)
        if season_meta is None:
            return None
        raw_all = _raw_predictions(conn, season)
        pitcher_rows = raw_all[raw_all["pitcher_id"] == pitcher_id]
        if pitcher_rows.empty:
            return None
        from analytics.stuff_model import _shrinkage_params, _shrunk_group_means, _to_stuff_plus
        k_by_type, fallback_k, type_means = _shrinkage_params(raw_all)
        g = _shrunk_group_means(pitcher_rows, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
        norm_mean = season_meta["norm_mean"]
        norm_std = season_meta["norm_std"]
        g["stuff_plus"] = _to_stuff_plus(g["shrunk_mean"], norm_mean, norm_std)
        import numpy as np
        return float(np.average(g["stuff_plus"], weights=g["n_pitches"]))

    if model == "tunneling":
        from analytics.tunneling_v2_model import _raw_predictions as tun_raw
        model_key = "tunneling_v3"
        season_meta = meta.get(model_key, {}).get(season_str)
        if season_meta is None:
            return None
        raw_all = tun_raw(conn, season)
        pitcher_rows = raw_all[raw_all["pitcher_id"] == pitcher_id]
        if pitcher_rows.empty:
            return None
        mean_xrv = float(pitcher_rows["xrv"].mean())
        norm_mean = season_meta["norm_mean"]
        norm_std = season_meta["norm_std"]
        return 100.0 - 10.0 * (mean_xrv - norm_mean) / norm_std if norm_std else None

    if model == "location":
        from analytics.location_v3_model import _raw_predictions as loc_raw, LOCATION_ACTIVE_VERSION
        model_key = f"location_{LOCATION_ACTIVE_VERSION}"
        season_meta = meta.get(model_key, {}).get(season_str)
        if season_meta is None:
            return None
        raw_all = loc_raw(conn, season)
        pitcher_rows = raw_all[raw_all["pitcher_id"] == pitcher_id]
        if pitcher_rows.empty:
            return None
        mean_xrv = float(pitcher_rows["xrv"].mean())
        norm_mean = season_meta["norm_mean"]
        norm_std = season_meta["norm_std"]
        return 100.0 - 10.0 * (mean_xrv - norm_mean) / norm_std if norm_std else None

    raise ValueError(f"Unknown model: {model!r}. Expected 'stuff', 'tunneling', or 'location'.")


def warm_cache_sync(full=True):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        _warm(conn, full)
    finally:
        conn.close()


def warm_cache_background():
    threading.Thread(target=warm_cache_sync, daemon=True).start()
