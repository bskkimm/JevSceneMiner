# Working on JevSceneMiner

JevSceneMiner converts recorded driving data into shared numerical facts and a
readable script, classifies the maneuver active at NOW, and merges timestamped
decisions into scenes. Only the script is sent to Jev.

## Find the relevant guide

- Read [Contributing](docs/CONTRIBUTING.md) for development checks and contracts.
- When adding a data source or diagnosing incorrect preprocessing, read
  [Adapt your driving data](docs/adapting-private-data.md) before changing code.
  Inspect representative source data and document units, clocks, frames and
  missing evidence. Ask about unresolved semantics before implementing affected
  conversions; continue independent work.
- Use [schema.md](docs/schema.md) as the sample/scene specification and
  [architecture.md](docs/architecture.md) for implementation locations.

## Preserve evidence and compatibility

- Prefer a source adapter under `src/jevsceneminer/inputs/` for raw-format changes.
  Use grouped modules and preserve legacy import bridges.
- Keep facts and readable evidence consistent. Missing signals, dimensions, map
  matches and stable actor identities remain unavailable; do not invent values.
- Distinguish mapped segment transitions/polygon overlap from verified physical
  lane ownership or maneuver labels. Verify frame and pose-reference assumptions.
- Preserve raw probabilities when merging scenes. Changes to the script,
  questions or model require compatible cache handling.
- Keep private recordings, credentials and generated runs out of Git. Use
  original synthetic fixtures for public tests. Do not send private recordings
  to external services or run paid inference without explicit authorization.

## Verify the change

After `uv sync --frozen`, run checks relevant to the change:

```bash
uv run pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
for file in tests/test_*.cjs; do node "$file" || exit 1; done
```

Tests must not make paid API calls. Add meaningful tests for behavior changes;
cover both built-in adapters when common preprocessing changes. Check a clean
wheel installation when packaging/resources change, as described in Contributing
and CI. Report checks run, source assumptions and remaining evidence limitations.
