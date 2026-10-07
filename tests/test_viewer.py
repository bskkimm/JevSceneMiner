import json

import pytest

from jevsceneminer.viewer import Site


@pytest.fixture()
def out(tmp_path):
    root = tmp_path / "out"
    for sub in ("meta", "steps", "scenes", "camera", "tags"):
        (root / sub).mkdir(parents=True)
    (root / "meta" / "s1.json").write_text(json.dumps({"date": "2021-01-01", "name": "s1", "start_ns": 0,
                                                      "end_ns": 10**10}))
    (root / "steps" / "s1.jsonl").write_text("\n".join(json.dumps({"t_ns": t * 10**9, "script": f"script at {t}"})
                                                       for t in range(10)))
    (root / "scenes" / "s1.json").write_text(json.dumps({"scenes": []}))
    (root / "camera" / "s1.json").write_text(json.dumps([[0, "log/CAM_F0/a.jpg"], [10**9, "../../etc/passwd"]]))
    sensors = tmp_path / "sensors" / "log" / "CAM_F0"
    sensors.mkdir(parents=True)
    (sensors / "a.jpg").write_bytes(b"jpg")
    return root


def test_session_script_and_camera(out):
    site = Site(out, None, out.parent / "sensors", {"jev": out / "scenes"})
    s = site.session("s1")
    assert len(s["steps"]) == 10 and "script" not in s["steps"][0]
    assert s["runs"]["jev"] == {"scenes": []} and s["gt"] is None and not s["gt_writable"]
    assert site.script("s1", int(3.4e9)) == "script at 3"
    assert site.camera_file("s1", 0).read_bytes() == b"jpg"
    assert site.camera_file("s1", 1) is None          # outside the sensor root
    assert site.bev("s1")["available"] is False        # not a nuPlan session


def test_gt_is_saved_only_with_a_gt_folder(out, tmp_path):
    with pytest.raises(PermissionError):
        Site(out, None, None, {}).save_gt("s1", {})
    path = Site(out, tmp_path / "gt", None, {}).save_gt("s1", {"scenes": []})
    assert json.loads(path.read_text()) == {"scenes": []}



def test_session_timeline_omits_large_audit_facts(tmp_path):
    import json
    from jevsceneminer.viewer import Site
    for folder in ('steps','meta'):(tmp_path/folder).mkdir()
    (tmp_path/'meta'/'s.json').write_text('{}')
    (tmp_path/'steps'/'s.jsonl').write_text(json.dumps({'t_ns':1,'script':'readable input',
        'facts':{'object_tracks':[{'observations':[{'time_s':0}]}]},'lateral':'keep_lane'})+'\n')
    site=Site(tmp_path,None,None,{})
    assert site.session('s')['steps']==[{'t_ns':1,'lateral':'keep_lane'}]
    assert site.script('s',1)=='readable input'



def test_script_cache_reuses_read_and_refreshes_after_dataset_change(out,monkeypatch):
    path=out/'steps'/'s1.jsonl'
    reads=[]
    original=type(path).open
    def opened(p,*args,**kwargs):
        mode=kwargs.get('mode',args[0] if args else 'r')
        if p==path and mode in ('r','rt'):reads.append(1)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(type(path),'open',opened)
    site=Site(out,None,None,{})
    assert site.script('s1',3_000_000_000)=='script at 3'
    assert site.script('s1',4_000_000_000)=='script at 4'
    assert len(reads)==1
    path.write_text(json.dumps({'t_ns':3_000_000_000,'script':'updated script'})+'\n')
    assert site.script('s1',3_000_000_000)=='updated script'
    assert len(reads)==2



def test_cached_script_does_not_wait_for_unrelated_map_work(out):
    import threading
    site=Site(out,None,None,{})
    assert site.script('s1',3_000_000_000)=='script at 3'
    completed=threading.Event()
    results=[]
    def request():
        results.append(site.script('s1',3_000_000_000))
        completed.set()
    # Background BEV builds hold the map cache lock; cached text must remain available.
    site._lock.acquire()
    worker=threading.Thread(target=request)
    try:
        worker.start()
        assert completed.wait(.5)
        assert results==['script at 3']
    finally:
        site._lock.release()
        worker.join(2)


def test_bev_keeps_native_box_dimensions_and_rear_axle_reference(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import shapely
    import jevsceneminer.nuplan as reader
    import jevsceneminer.nuplan_map as maps
    from jevsceneminer.bag import DetectedObject
    ego=SimpleNamespace(t=np.array([0,100_000_000]), x=np.array([100.,101.]),
        y=np.array([200.,200.]),yaw=np.array([0.,0.]))
    obj=DetectedObject('car',105.,202.,0.,0.,1.,length_m=5.95,width_m=2.16)
    session=SimpleNamespace(ego=ego,objects=[(0,[obj])])
    monkeypatch.setattr(reader,'read_nuplan',lambda *a,**k:SimpleNamespace(session=session))
    monkeypatch.setattr(maps,'load_nuplan_map',lambda *a:SimpleNamespace(
        polygons={1:shapely.box(99,198,110,204)},lanes={1:SimpleNamespace(is_intersection=False)}))
    bev=Site._build_bev({'sources':['log.db'],'map':'map.gpkg'})
    assert bev['objects'][0][1][0][5:]==[5.95,2.16]
    assert bev['ego_vehicle']=={'length_m':5.176,'width_m':2.297,'rear_axle_to_center_m':1.461}


def test_local_bev_alignment_is_scoped_and_preserves_original_lanes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import shapely
    import jevsceneminer.nuplan as reader
    import jevsceneminer.nuplan_map as maps
    ego=SimpleNamespace(t=np.array([0,100_000_000]), x=np.array([100.,101.]),
        y=np.array([200.,200.]),yaw=np.array([0.,0.]))
    monkeypatch.setattr(reader,'read_nuplan',lambda *a,**k:SimpleNamespace(session=SimpleNamespace(ego=ego,objects=[])))
    monkeypatch.setattr(maps,'load_nuplan_map',lambda *a:SimpleNamespace(
        polygons={1:shapely.box(99,198,110,201),2:shapely.box(99,201,110,204)},
        lanes={i:SimpleNamespace(is_intersection=False) for i in [1,2]}))
    patch=tmp_path/'alignment.json'
    doc={'format':'jsm-bev-alignment-1','source_map':str(tmp_path/'map.gpkg'),
         'source_logs':['log.db'],'description':'Camera-aligned local lanes (approx.)',
         'lanes':[{'id':1,'polygon_xy':[[99,199],[110,199],[110,202],[99,202],[99,199]]}]}
    patch.write_text(json.dumps(doc))
    meta={'sources':['log.db'],'map':str(tmp_path/'map.gpkg'),'bev_alignment':str(patch)}
    b=Site._build_bev(meta)
    assert b['lanes'][0]['xy']!=b['original_lanes'][0]['xy']
    assert b['lanes'][1]==b['original_lanes'][1]
    assert b['map_alignment']['description']==doc['description']
    with pytest.raises(ValueError,match='source log'):
        Site._build_bev(dict(meta,sources=['another.db']))
    with pytest.raises(ValueError,match='source map'):
        Site._build_bev(dict(meta,map=str(tmp_path/'another.gpkg')))
    doc['lanes'][0]['polygon_xy']=[[99,199],[110,202],[99,202],[110,199],[99,199]]
    patch.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='invalid polygon'):
        Site._build_bev(meta)


def test_bev_cache_changes_when_local_alignment_changes(tmp_path):
    source=tmp_path/'log.db';source.write_bytes(b'log')
    mapfile=tmp_path/'map.gpkg';mapfile.write_bytes(b'map')
    patch=tmp_path/'alignment.json';patch.write_text('{}')
    meta={'sources':[str(source)],'map':str(mapfile)}
    original=Site.bev_key(meta)
    meta['bev_alignment']=str(patch)
    before=Site.bev_key(meta)
    assert before!=original
    patch.write_text('{"updated":true}')
    assert Site.bev_key(meta)!=before
