"""Entries keyed only on finished seasons live a week; current-season and season-less entries keep the
cache's own TTL."""
import time

from api import cache as C
from config.model_policy import CURRENT_SEASON


def _store(wrapper):
    for cell in wrapper.__closure__:
        c = cell.cell_contents
        if isinstance(c, dict) and c and all(isinstance(v, tuple) for v in c.values()):
            return c
    raise AssertionError("store not found")


def test_past_season_entries_live_longer():
    @C.ttl_cache(ttl_seconds=60)
    def build(conn, *key):
        return key

    build(None, CURRENT_SEASON - 1)          # finished season
    build(None, CURRENT_SEASON)              # current season
    build(None, 543037)                      # no season in the key (a player id)
    build(None, CURRENT_SEASON - 1, CURRENT_SEASON)  # mixes a finished and the current season
    life = {k[0]: exp - time.monotonic() for k, (_, exp) in _store(build).items()}
    assert life[(CURRENT_SEASON - 1,)] > C.PAST_SEASON_TTL_SECONDS - 5
    assert 55 < life[(CURRENT_SEASON,)] <= 60
    assert 55 < life[(543037,)] <= 60
    assert 55 < life[(CURRENT_SEASON - 1, CURRENT_SEASON)] <= 60


def test_clear_modules_clears_only_the_named_modules():
    @C.ttl_cache()
    def data_page(conn, key):
        return key

    data_page.__module__ = "api.routers.prospects_probe"
    other = C.ttl_cache()(lambda conn, key: key)
    other.__module__ = "api.routers.standings_probe"
    data_page(None, 1)
    other(None, 1)
    C.clear_modules(("api.routers.prospects",))
    assert not _store_or_empty(data_page)
    assert _store_or_empty(other)


def _store_or_empty(fn):
    for cell in fn.__closure__:
        c = cell.cell_contents
        if isinstance(c, dict) and (not c or all(isinstance(v, tuple) for v in c.values())):
            if c or "store" in str(fn.__code__.co_freevars):
                return c
    return {}
