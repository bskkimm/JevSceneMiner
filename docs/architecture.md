# Repository layout

The tracked tree is small and separates implementation, tests, examples and documentation.
Local datasets, run outputs, API keys and internal notes are ignored; they should stay outside the publication tree.

## Current tree

```text
JevSceneMiner/
├── jevsceneminer/       Python package: adapters, evidence, inference, merging
│   ├── presets/        Explicit run configurations
│   └── viewer/         Browser UI and geometry helpers
├── tests/              Python and browser-logic tests
├── examples/           Original synthetic pipeline and sample artifacts
├── docs/               Usage, schema, review, evaluation and demo notes
│   └── assets/         Small documentation illustrations
├── tools/              Packaging and publication checks
├── .github/            CI and issue/PR templates
├── labels.yaml         Default Jev questions
├── labels.compact.yaml Experimental compact-input questions
├── pyproject.toml
└── uv.lock
```

The root is already reasonably clean. The main maintainability issue is the flat Python package: source adapters, evidence calculations and display geometry are intermixed.
There is no need to add a directory for every file or change working imports just for appearance.

## Proposed next cleanup

This is a proposal, not the tree implemented by the README update:

```text
JevSceneMiner/
├── src/jevsceneminer/
│   ├── inputs/         Rosbag and nuPlan adapters
│   ├── evidence/       Motion, lane/body and object evidence; script rendering
│   ├── inference/      Jev client, cache and run orchestration
│   ├── scenes/         Merging, phases and evaluation
│   ├── viewer/         Server, camera projection and browser assets
│   ├── presets/
│   └── cli.py
├── tests/              Mirrors the stable component boundaries
├── examples/
├── docs/
├── tools/
└── .github/
```

Do this as one separate refactoring PR after freezing module boundaries.
Preserve the CLI, sample/scene schemas and packaged resources; keep compatibility wrappers for existing import paths where needed.
Verify both source adapters and a clean wheel installation before merging.

## Small, useful commits

1. **Presentation:** concise README, a real demo and linked detailed guides.
2. **Documentation assets:** keep large MP4s as GitHub attachments rather than Git blobs; retain only small assets with a clear use.
3. **Package organization:** apply the proposed module grouping with import and packaging checks.
4. **Evaluation:** add a reproducible held-out benchmark when independent GT and distributable evidence are available.

The first two belong to this documentation update. The package move and benchmark are future work.
