# Contributing

Small, reviewable PRs are welcome. Explain the concrete behavior before and after, and include relevant validation.
Discuss major schema or label-policy changes in an issue first.

## Development

Linux with Python 3.10 or 3.11 and Node.js 22 is tested in CI.

```bash
uv sync --frozen
uv run pre-commit install
uv run pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
for file in tests/test_*.cjs; do node "$file"; done
uv build --wheel
```

Disabling automatic pytest plugin loading avoids unrelated ROS workspace plugins.
Tests and synthetic demos do not require an API key and must not make paid API calls.
CI also installs a wheel into a separate environment and checks packaged labels, preset and viewer files.

Implementation lives in `src/jevsceneminer/`; use its grouped modules for new code.
Keep legacy import bridges intact and verify wheel resources after layout changes.
See [repository layout](docs/architecture.md).

## Contracts to preserve

- Missing map matches, dimensions, signals or actor identities remain unknown.
- Facts and script measurements share a source and explicit intervals.
- Lane polygon overlap is nonexclusive; physical ownership requires additional evidence.
- Scene phases are contiguous and end at the parent scene end.
- Raw probabilities remain raw when merging or extending scenes.
- Changed questions/model/script invalidate classifier cache keys.
- Manual evidence annotations stay opt-in with source identity and provenance.
- Coordinate transforms must distinguish pose reference, body center and camera capture time.

Use meaningful regression tests for new behavior and failures. Include both source adapters when changing common preprocessing.
Label changes need held-out GT evaluation, not only a curated example.
See [schema](docs/schema.md), [review](docs/review.md), and [evaluation](docs/evaluation.md).

## Public files

Do not commit keys, raw logs/maps/images, output folders or internal notes.
Use original synthetic fixtures when possible; external assets retain their source terms.
`tools/check_publication.py` checks tracked files for common accidental data/secret inclusions.
An API budget, inference run and redistribution rights require deliberate decisions outside the test suite.
