const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../src/jevsceneminer/viewer/static/editor.js');
const bounds = [0, 20];
const base = () => [
  {start: 0, end: 5, lateral: 'keep_lane', longitudinal: 'cruising'},
  {start: 5, end: 10, lateral: 'turn_right', longitudinal: 'decelerating'},
  {start: 10, end: 20, lateral: 'keep_lane', longitudinal: 'cruising'},
];

test('editing a shared boundary updates the neighbor and preserves labels', () => {
  const scenes = base();
  const edited = E.update(scenes, 1, {start: 6, end: 12, lateral: 'lane_change_right'}, bounds);
  assert.equal(edited[0].end, 6);
  assert.equal(edited[1].start, 6);
  assert.equal(edited[1].end, 12);
  assert.equal(edited[2].start, 12);
  assert.equal(edited[1].lateral, 'lane_change_right');
  assert.equal(edited[1].longitudinal, 'decelerating');
  assert.deepEqual(scenes, base()); // undo needs the original to remain intact
});

test('boundary edits cannot erase a neighboring scene or exceed the session', () => {
  const edited = E.update(base(), 1, {start: -10, end: 100}, bounds);
  assert.equal(edited[0].start, 0);
  assert.ok(edited[0].end - edited[0].start >= 0.099);
  assert.ok(edited[2].end - edited[2].start >= 0.099);
  assert.equal(edited[2].end, 20);
  assert.equal(edited[0].end, edited[1].start);
  assert.equal(edited[1].end, edited[2].start);
  assert.throws(() => E.update(base(), 1, {start: NaN}, bounds), /finite/);
});

test('adding a manual label creates GT from empty and replaces only the painted interval', () => {
  const label = {lateral: 'lane_change_left', longitudinal: 'accelerating'};
  const empty = E.paint([], 2, 4, label, bounds);
  assert.deepEqual(empty, [{start: 2, end: 4, ...label}]);
  const painted = E.paint(base(), 3, 7, label, bounds);
  assert.deepEqual(painted.map(s => [s.start, s.end, s.lateral]), [
    [0, 3, 'keep_lane'], [3, 7, 'lane_change_left'], [7, 10, 'turn_right'], [10, 20, 'keep_lane'],
  ]);
});

test('split preserves both labels and merge uses the left label', () => {
  const split = E.split(base(), 1, 7);
  assert.deepEqual(split.map(s => [s.start, s.end]), [[0, 5], [5, 7], [7, 10], [10, 20]]);
  assert.equal(split[2].longitudinal, 'decelerating');
  assert.deepEqual(E.mergeLeft(split, 2), base());
  assert.deepEqual(E.split(base(), 1, 5), base());
});

test('delete gives the interval to an adjacent scene; last scene can be removed', () => {
  const deleted = E.remove(base(), 1);
  assert.deepEqual(deleted.map(s => [s.start, s.end, s.lateral]), [[0, 10, 'keep_lane'], [10, 20, 'keep_lane']]);
  assert.deepEqual(E.remove([{start: 2, end: 4, lateral: 'turn_left', longitudinal: 'cruising'}], 0), []);
});

test('edits preserve unlabeled gaps and never merge scenes across a gap', () => {
  const scenes = [{...base()[0], end: 3}, {...base()[1], start: 5}];
  const edited = E.update(scenes, 1, {start: 1}, bounds);
  assert.equal(edited[0].end, 3);
  assert.equal(edited[1].start, 3);
  assert.deepEqual(E.mergeLeft(scenes, 1), scenes);
  assert.deepEqual(E.remove(scenes, 1), [scenes[0]]);
});


test('Apply can move a scene beyond its old end when both requested times are valid', () => {
  const edited = E.update(base(), 1, {start: 12, end: 15}, bounds);
  assert.equal(edited[1].start, 12);
  assert.equal(edited[1].end, 15);
  assert.equal(edited[0].end, 12);
  assert.equal(edited[2].start, 15);
});

test('painting lateral labels preserves every underlying speed transition', () => {
  const painted = E.paintRow(base(), 4, 7, {lateral: 'lane_change_left'}, bounds);
  assert.deepEqual(painted.map(s => [s.start, s.end, s.lateral, s.longitudinal]), [
    [0, 4, 'keep_lane', 'cruising'], [4, 5, 'lane_change_left', 'cruising'],
    [5, 7, 'lane_change_left', 'decelerating'], [7, 10, 'turn_right', 'decelerating'],
    [10, 20, 'keep_lane', 'cruising'],
  ]);
});

test('painting speed labels preserves lane transitions and can label an empty interval', () => {
  const painted = E.paintRow(base(), 4, 7, {longitudinal: 'accelerating'}, bounds);
  assert.deepEqual(painted.filter(s => s.start >= 4 && s.end <= 7).map(s => [s.lateral, s.longitudinal]),
    [['keep_lane', 'accelerating'], ['turn_right', 'accelerating']]);
  assert.deepEqual(E.paintRow([], 2, 4, {lateral: 'turn_left'}, bounds),
    [{start: 2, end: 4, lateral: 'turn_left', longitudinal: 'cruising'}]);
});


test('dragging only one boundary leaves the opposite edge fixed', () => {
  const end = E.update(base(), 1, {end: 3}, bounds);
  assert.equal(end[1].start, 5);
  assert.equal(end[1].end, 5.1);
  assert.equal(end[0].end, 5);
  const start = E.update(base(), 1, {start: 12}, bounds);
  assert.equal(start[1].start, 9.9);
  assert.equal(start[1].end, 10);
  assert.equal(start[2].start, 10);
});

test('row painting includes short fragments on both sides of a label boundary', () => {
  const painted = E.paintRow(base(), 4.95, 7, {lateral: 'lane_change_left'}, bounds);
  assert.deepEqual(painted.filter(s => s.start >= 4.95 && s.end <= 7).map(s =>
    [s.start, s.end, s.lateral, s.longitudinal]),
    [[4.95, 5, 'lane_change_left', 'cruising'], [5, 7, 'lane_change_left', 'decelerating']]);
});

const nested = () => [
  {start:0,end:5,lateral:'keep_lane',phases:[{end:2,decision:'accelerating'},{end:5,decision:'cruising'}]},
  {start:5,end:10,lateral:'turn_right',phases:[{end:8,decision:'decelerating'},{end:10,decision:'stopped'}]},
];
function covered(scenes) {
  for(const s of scenes) {let start=s.start; for(const p of s.phases){assert.ok(p.end>start);start=p.end;}assert.equal(start,s.end);}
}
test('legacy adjacent same driving decisions become one parent, preserving gaps',()=>{
  const scenes=E.fromDocument({scenes:[{start_ns:'0',end_ns:'2000000000',lateral:'keep_lane',longitudinal:'accelerating'},
    {start_ns:'2000000000',end_ns:'5000000000',lateral:'keep_lane',longitudinal:'cruising'},
    {start_ns:'6000000000',end_ns:'7000000000',lateral:'keep_lane',longitudinal:'stopped'}]});
  assert.equal(scenes.length,2);assert.deepEqual(scenes[0].phases,[{end:2,decision:'accelerating'},{end:5,decision:'cruising'}]);covered(scenes);
});
test('nested shared resize clips and extends edge speed phases without mutating undo',()=>{
  const original=nested(); const changed=E.update(original,0,{end:9},[0,10]);covered(changed);
  assert.deepEqual(changed[0].phases,[{end:2,decision:'accelerating'},{end:9,decision:'cruising'}]);
  assert.deepEqual(changed[1].phases,[{end:10,decision:'stopped'}]);assert.deepEqual(original,nested());
  const shrunk=E.update(original,0,{end:1},[0,10]);covered(shrunk);
  assert.deepEqual(shrunk[0].phases,[{end:1,decision:'accelerating'}]);
});
test('nested split and merge preserve all speed transitions and left driving decision',()=>{
  const scenes=nested();const split=E.split(scenes,0,3);covered(split);
  assert.deepEqual(split[1].phases,[{end:5,decision:'cruising'}]);
  const merged=E.mergeLeft(scenes,1);covered(merged);assert.equal(merged.length,1);assert.equal(merged[0].lateral,'keep_lane');assert.equal(merged[0].phases.length,4);
  assert.deepEqual(scenes,nested());
});
test('nested lateral paint creates a single parent retaining speed phases',()=>{
  const original=nested(), painted=E.paintRow(original,1,9,{lateral:'lane_change_left'},[0,10]);covered(painted);
  const parent=painted.find(s=>s.start===1);assert.equal(parent.end,9);
  assert.deepEqual(parent.phases,[{end:2,decision:'accelerating'},{end:5,decision:'cruising'},{end:8,decision:'decelerating'},{end:9,decision:'stopped'}]);
  parent.phases[0].decision='stopped';assert.deepEqual(original,nested());
});
test('phase label, split, boundary and delete remain inside parent',()=>{
  const scenes=nested(); const split=E.phaseSplit(scenes,0,3);covered(split);assert.equal(split[0].phases.length,3);
  const edit=E.phaseUpdate(split,0,1,{end:4,decision:'decelerating'});covered(edit);assert.equal(edit[0].phases[1].decision,'decelerating');
  assert.throws(()=>E.phaseUpdate(edit,0,0,{end:4}),/strictly/);
  assert.equal(E.phaseUpdate(edit,0,2,{end:100})[0].phases[2].end,5);
  covered(E.phaseRemove(edit,0,2));assert.deepEqual(scenes,nested());
});
test('nested scene deletion extends the receiving edge phase',()=>{
  const deleted=E.remove(nested(),1);covered(deleted);assert.deepEqual(deleted[0].phases,[{end:2,decision:'accelerating'},{end:10,decision:'cruising'}]);
});
test('painting preserves deep copies of unaffected scenes for undo',()=>{
  const original=nested();const painted=E.paint(original,1,2,{lateral:'turn_left',phases:[{end:2,decision:'unknown'}]},[0,10]);
  painted.at(-1).phases[0].decision='accelerating';assert.deepEqual(original,nested());
});
test('new documents preserve separate parents and missing legacy speed is unknown',()=>{
  const scenes=E.fromDocument({scenes:[{start_ns:'0',end_ns:'1000000000',driving_decision:'keep_lane',longitudinal_phases:[{end_ns:'1000000000',decision:'unknown'}]},
    {start_ns:'1000000000',end_ns:'2000000000',driving_decision:'keep_lane',longitudinal_phases:[{end_ns:'2000000000',decision:'cruising'}]}]});
  assert.equal(scenes.length,2);covered(scenes);
  assert.equal(E.fromDocument({scenes:[{start_ns:'0',end_ns:'1000000000',lateral:'keep_lane'}]})[0].phases[0].decision,'unknown');
});
test('painting lateral across a data gap marks missing speed unknown',()=>{
  const scenes=nested();scenes[1].start=7;
  const painted=E.paintRow(scenes,1,9,{lateral:'turn_left'},[0,10]);covered(painted);
  assert.ok(painted.find(s=>s.start===1).phases.some(p=>p.end===7 && p.decision==='unknown'));
});
