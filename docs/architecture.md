# Repository layout

Implementation lives under `src/`, separate from tests, examples and documentation.
Local datasets, outputs, API keys and internal notes remain ignored.

```text
JevSceneMiner/
├── src/jevsceneminer/
│   ├── inputs/         Rosbag, nuPlan and native-map adapters
│   ├── evidence/       Motion, lane/body and object evidence; input scripts
│   ├── inference/      Jev questions, client and answer cache
│   ├── scenes/         Lateral merging, speed phases, rules and evaluation
│   ├── viewer/         Server and camera/height projection
│   │   └── static/     HTML and browser geometry/editor helpers
│   ├── presets/        Explicit run configurations
│   └── cli.py
├── tests/              Python and browser-logic contracts
├── examples/           Original synthetic pipeline and sample artifacts
├── benchmarks/         Small synthetic, hash-locked scoring contract
├── docs/               Usage, schema, review, evaluation and demo notes
│   └── assets/         Small documentation previews
├── tools/              Packaging and publication checks
├── .github/            CI and issue/PR templates
├── labels.yaml
├── labels.compact.yaml
├── pyproject.toml
└── uv.lock
```

## Import compatibility

Use the grouped modules for new implementation, such as `jevsceneminer.inputs.nuplan` or `jevsceneminer.evidence.script`.
The former top-level imports remain small compatibility bridges to the same module objects.
There are no duplicate implementations: source types and patches retain their identity across both import paths.
`jevsceneminer.scenes` keeps its original merge API, and `jevsceneminer.viewer` keeps its server API, including explicit `PAGE` overrides.
The older `python -m jevsceneminer.compact` entry point remains usable.

The CLI, data schemas, exact questions, cache keys and merge behavior are unchanged by this move.
Labels, presets and browser assets are checked after installing a wheel outside the source checkout.
Both synthetic adapters are also compared before and after migration.

## Keep the tree small

Keep large MP4s as GitHub attachments, raw data and derived runs outside Git, and only useful documentation assets under `docs/assets/`.
The short README preview is a single optimized GIF; the full video is hosted separately.
Avoid adding folders for speculative features. Evaluation should use saved results and independent, locked GT rather than a curated showcase.
