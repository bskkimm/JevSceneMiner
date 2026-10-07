"""Exercise both source adapters through canonical preprocessing and cached inference."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pipeline_demo',Path(__file__).resolve().parents[1]/'examples/pipeline_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.parametrize('source', ['rosbag', 'nuplan'])
def test_synthetic_source_to_structured_sample_and_nested_scene(tmp_path,source):
    out = demo.build_demo(tmp_path/source,source)
    row = json.loads((out/'sample.json').read_text())
    assert row['facts']['schema_version']=='jsm-sample-1.1'
    assert len(row['facts']['ego_samples'])==26
    assert row['facts']['ego_samples'][10]['time_s']==0
    assert row['lateral']=='lane_change_right' and row['longitudinal']=='cruising'
    assert json.loads((out/'demo.json').read_text())['api_calls']==0
    meta = json.loads(next((out/'meta').glob('*.json')).read_text())
    assert meta['has_indicator']==(source=='rosbag')
    scenes=json.loads(next((out/'scenes').glob('*.json')).read_text())['scenes']
    assert any(s['driving_decision']=='lane_change_right' for s in scenes)
    assert all(s['longitudinal_phases'][-1]['end_ns']==s['end_ns'] for s in scenes)
    assert json.loads(next((out/'runtime').glob('*.json')).read_text())['fresh_answers']==0
