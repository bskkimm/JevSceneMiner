"""Publication contracts: portable settings and bounded inference, without API calls."""
import json
from types import SimpleNamespace

import pytest


def test_runtime_counts_each_fresh_response_for_repeated_scripts(tmp_path, monkeypatch):
    from jevsceneminer import cli
    from jevsceneminer.jev import JevClassifier
    from test_jev import FakeClient, LABELS
    from types import SimpleNamespace
    import json

    (tmp_path/'steps').mkdir()
    scripts={i*10**9:'identical stopped context' for i in range(3)}
    (tmp_path/'steps'/'date_session.jsonl').write_text(''.join(
        json.dumps(dict(t_ns=t,script=text))+'\n' for t,text in scripts.items()))
    meta=dict(date='date',name='session',start_ns=0,end_ns=3*10**9,step_s=1)
    args=SimpleNamespace(model='test',workers=2,require_indicator=False,
        min_scene=2,min_phase=1,min_prob=0,maneuver_tail=0)
    client=FakeClient()
    monkeypatch.setattr(cli,'JevClassifier',lambda labels,cache,**kwargs:
        JevClassifier(labels,cache,client_factory=lambda:client,log=lambda _:None,**kwargs))
    cli._classify(tmp_path,meta,scripts,LABELS,args)
    runtime=json.loads((tmp_path/'runtime'/'date_session.json').read_text())
    assert runtime['fresh_answers']==client.calls
    assert runtime['input_tokens']==client.calls*100

from jevsceneminer import cli
from jevsceneminer.jev import Answer, cache_key, load_labels


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.setattr(cli, 'load_dotenv', lambda: None)
    monkeypatch.setenv('TYPESAFE_API_KEY', 'test-placeholder')


def test_preferred_preset_and_explicit_override(no_credentials, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, 'cmd_run', lambda args: captured.append(args) or 0)
    assert cli.main(['run', '--nuplan', 'example.db', '--out', 'unused', '--dry-run',
                     '--preset', 'nuplan-1hz', '--future', '12', '--step', '0.5']) == 0
    args = captured[0]
    assert (args.past, args.future, args.step, args.table_step) == (10, 12, .5, 1)
    assert args.model == 'jev-1.13.0' and args.maneuver_tail == 2
    assert args.camera_height_reference == 'stationary_landmarks'
    assert args.camera_ground_offset == .24


def test_preset_requires_nuplan_and_rejects_bad_projection_offset(no_credentials, monkeypatch):
    monkeypatch.setattr(cli, 'cmd_run', lambda _: pytest.fail('invalid input reached execution'))
    with pytest.raises(SystemExit):
        cli.main(['run', 'bag', '--map', 'map.osm', '--out', 'unused', '--dry-run', '--preset', 'nuplan-1hz'])
    with pytest.raises(SystemExit):
        cli.main(['run', '--nuplan', 'example.db', '--out', 'unused', '--dry-run', '--camera-ground-offset', 'nan'])


def test_default_cadence_stays_compatible(no_credentials, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, 'cmd_run', lambda args: captured.append(args) or 0)
    assert cli.main(['run', '--nuplan', 'example.db', '--out', 'unused', '--dry-run']) == 0
    assert (captured[0].past, captured[0].future, captured[0].step, captured[0].table_step) == (10, 10, .5, .5)


def test_preparation_records_projection_and_model_settings(no_credentials, monkeypatch, tmp_path, lanemap, lane_change_session):
    def prepare(args):
        meta, _ = cli._prepare(lane_change_session, lanemap, 'right', {}, args, tmp_path)
        assert meta['camera_ground_offset_m'] == .24
        assert meta['camera_height_reference'] == 'stationary_landmarks'
        assert meta['inference_model'] == 'jev-1.13.0'
        assert (meta['past_s'], meta['future_s'], meta['table_step_s']) == (10, 15, 1)
        return 0
    monkeypatch.setattr(cli, 'cmd_run', prepare)
    assert cli.main(['run', '--nuplan', 'example.db', '--out', str(tmp_path), '--dry-run', '--preset', 'nuplan-1hz']) == 0


def seed_run(tmp_path):
    labels = load_labels(cli.REPO_ROOT / 'labels.yaml', traffic_side='right', has_indicator=False)
    meta = dict(date='2026-01-01', name='synthetic', start_ns=10**9, end_ns=6*10**9,
                step_s=1, traffic_side='right', has_indicator=False, inference_model='jev-test')
    for folder in ['meta', 'steps', 'cache']:
        (tmp_path / folder).mkdir()
    sid = meta['date']+'_'+meta['name']
    (tmp_path/'meta'/f'{sid}.json').write_text(json.dumps(meta))
    rows = []
    for i in range(5):
        text = f'synthetic observed state {i}'
        rows.append(dict(t_ns=(i+1)*10**9, script=text, facts={'source': 'synthetic', 'index': i},
                         lateral='keep_lane', lateral_probs={'keep_lane': 1},
                         longitudinal='cruising', longitudinal_probs={'cruising': 1},
                         model='jev-test', input_tokens=10, cache_key=cache_key('jev-test', labels, text)))
    path = tmp_path/'steps'/f'{sid}.jsonl'
    original = [json.dumps(row, indent=None)+'  \n' for row in rows]
    path.write_text(''.join(original))
    return path, original, rows


def install_classifier(monkeypatch):
    calls = []
    class ObservedClassifier:
        def __init__(self, labels, cache, model=None, workers=8):
            self.model = model
        def classify(self, scripts):
            calls.append(dict(scripts))
            return {t: Answer('keep_lane', {'keep_lane': 1}, 'accelerating', {'accelerating': 1},
                              self.model, 20) for t in scripts}
    monkeypatch.setattr(cli, 'JevClassifier', ObservedClassifier)
    return calls


def test_ranged_inference_preserves_outside_lines_and_merges_all_answers(no_credentials, monkeypatch, tmp_path):
    path, original, rows = seed_run(tmp_path)
    calls = install_classifier(monkeypatch)
    assert cli.main(['classify', str(tmp_path), '--from-s', '1', '--to-s', '3']) == 0
    assert list(calls[0]) == [2*10**9, 3*10**9]
    after = path.read_text().splitlines(keepends=True)
    assert len(after) == 5
    assert [after[i] for i in [0, 3, 4]] == [original[i] for i in [0, 3, 4]]
    assert all(json.loads(after[i])['facts'] == rows[i]['facts'] for i in range(5))
    scene = json.loads(next((tmp_path/'scenes').glob('*.json')).read_text())
    assert scene['scenes'][0]['start_ns'] == str(10**9)
    # Inference cells stop half a step after the last observed NOW, not at an unsupported log tail.
    assert scene['scenes'][-1]['end_ns'] == str(5*10**9+500_000_000)
    assert scene['generator']['reclassified_samples'] == 2


@pytest.mark.parametrize('flags', [['--from-s', '1'], ['--from-s', '3', '--to-s', '1'],
                                  ['--from-s', 'nan', '--to-s', '3'], ['--from-s', '-1', '--to-s', '3']])
def test_invalid_ranges_fail_before_inference(no_credentials, monkeypatch, tmp_path, flags):
    seed_run(tmp_path)
    calls = install_classifier(monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(['classify', str(tmp_path), *flags])
    assert calls == []


def test_rerun_rejects_incompatible_outside_questions_before_calls(no_credentials, monkeypatch, tmp_path):
    path, original, _ = seed_run(tmp_path)
    calls = install_classifier(monkeypatch)
    with pytest.raises(ValueError, match='outside.*(question|cache)'):
        cli.main(['classify', str(tmp_path), '--model', 'different-model', '--from-s', '1', '--to-s', '3'])
    assert calls == [] and path.read_text() == ''.join(original)


def test_empty_range_is_a_noop(no_credentials, monkeypatch, tmp_path):
    path, original, _ = seed_run(tmp_path)
    calls = install_classifier(monkeypatch)
    assert cli.main(['classify', str(tmp_path), '--from-s', '20', '--to-s', '21']) == 0
    assert calls == [] and path.read_text() == ''.join(original)
