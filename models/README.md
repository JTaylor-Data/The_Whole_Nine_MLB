# models/

LightGBM booster files (.txt) for the active ML models. Marked as binary in `.gitattributes`.

## Current (live)

- `stuff_plus_v4_lgbm.txt` — Stuff+ v4 booster (ACTIVE)
- `stuff_plus_v4_lgbm_meta.json` — Stuff+ v4 metadata
- `location_plus_v3_*.txt` / related files — Location+ v3 booster (ACTIVE)
- `location_plus_v3_bip.txt` — Location+ v3 BIP run-value table (ACTIVE; read at startup)

## Archive (unproven)

Moved to the local `Archive/models_archive/` (not in the repo): Location+ v2 boosters, Deception+ v1 booster, Deception+ v2 fold models. The Deception+ v2b fold models stay here: they are the only copy of the live model and are needed to score a new season.
See `docs/methodology/06_archive_policy.md`.
