# Evidence review and reruns

Preserve the original run and inspect camera, native map, source poses and tracks before changing evidence.
Display-only alignment or height offsets do not correct inference inputs.
Physical-lane annotations change input evidence and stay explicitly identified.

## Physical-lane annotation

`lane_review.apply_lane_review(row, review)` returns a deep copy.
Intervals use elapsed session seconds and affect observation times, including context in neighboring NOW windows.

```python
from jevsceneminer.lane_review import apply_lane_review

review = {
    "session_id": "your-session-id",
    "start_ns": 1767000000000000000,
    "from_s": 100.0, "to_s": 118.0, "margin_s": 1.0,
    "chain_id": "reviewed_road",
    "source": "camera-reviewed interval; reviewer/date/reference"
}
updated = apply_lane_review(original_row, review)
```

The core states the same manually reviewed physical chain and 100% ownership: an annotation, not a measured polygon fraction.
Margins have unknown lane evidence.
Contradictory native matches, occupancy claims, map signals and lane events are removed in the affected interval.
Native motion and body gaps remain; invalid map-based object relationships become unknown.
No helper assigns a predicted maneuver, and no Boston interval is applied automatically.

This function neither writes files nor checks a run's session ID.
Callers must select the exact session, validate interval/provenance, preserve native rows, and write returned rows into a copied run.
The compact generator checks session identity when supplied with `--review`.

## Missing actor history

`moving_history.derive_observations` derives relative motion, body gaps and connector overlap from explicit source footprints and interpolated ego poses.
`moving_history.augment_sample` adds an identified actor to a copied sample and records source identity, any replaced track, connector and reporting tolerance.
These offline helpers supply evidence, not corrected predictions.

Connector overlap is nonexclusive occupancy, not avoidance intent.
Check path obstruction and ego response. Preserve observation timestamps and stable identity; rounded positions cannot safely identify actors.

## Selected NOWs

```bash
uv run jevsceneminer classify out/reviewed \
  --from-s 100 --to-s 120 --maneuver-tail 2
```

Start is inclusive, end exclusive, measured from session start.
It applies to every session in the folder; use a folder containing only the desired session.
Only selected NOWs are classified; their context stays the prepared full window.
Outside raw rows remain byte-for-byte unchanged. All answers rebuild scenes, so nearby merged boundaries may change.

Outside answers must match the exact model and resolved questions.
Changing labels/model requires a full pass; changed scripts get new keys and identical scripts reuse cache.
Completed cache entries survive a Jev error for retry.
Runtime files describe the latest invocation, not lifetime billing.

## Compact experiment and evaluation

```bash
uv run python -m jevsceneminer.compact out/native --out out/compact
uv run jevsceneminer classify out/compact --labels labels.compact.yaml
```

This optional renderer reduces repeated prose with separate questions and outputs.
It is not the default verbose pipeline.
Supply `--review annotation.json` only after source review.
Measure regressions on unchanged held-out GT before adopting new input formats.

Viewer Save writes separate GT; it neither trains Jev nor changes predictions.
Lock evaluation GT before tuning annotations, label text or merge parameters.
`classify --strip-map` uses a separately prepared map-independent actor shortlist and removes map text.
It requires a complete pass; partial ablations would mix incompatible inputs.
