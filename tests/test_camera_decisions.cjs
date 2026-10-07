const {test}=require('node:test');
const assert=require('node:assert/strict');
const D=require('../src/jevsceneminer/viewer/static/camera_decisions.js');
const scenes=[{start:100,end:110,lateral:'turn_right',phases:[{end:104,decision:'decelerating'},{end:110,decision:'accelerating'}]},
 {start:110,end:120,lateral:'keep_lane',phases:[{end:120,decision:'cruising'}]}];
const steps=[{t_ns:102e9,lateral_probs:{keep_lane:.8,turn_right:.2},longitudinal_probs:{accelerating:.1,decelerating:.9}},
 {t_ns:106e9,lateral_probs:{keep_lane:.1,turn_right:.9},longitudinal_probs:{accelerating:.8,decelerating:.2}},
 {t_ns:110e9,lateral_probs:{keep_lane:1,turn_right:0},longitudinal_probs:{accelerating:0,cruising:1}}];
test('phase boundary belongs to the next phase; scene end belongs to next scene',()=>{
 assert.equal(D.at(scenes,104).phase.decision,'accelerating');
 assert.equal(D.at(scenes,110).scene.lateral,'keep_lane'); assert.equal(D.at(scenes,120),null);
});
test('merged mean support uses its own scene/phase interval, independent of raw winner',()=>{
 const v=D.view(scenes,steps,102);assert.equal(v.rawLateral[0][0],'keep_lane');assert.equal(v.lateral,'turn_right');
 assert.ok(Math.abs(v.lateralSupport-.55)<1e-10); assert.equal(v.longitudinalSupport,.9);
 assert.equal(D.view(scenes,steps,106).longitudinalSupport,.8);
 assert.equal(D.view(scenes,[],102).lateralSupport,null);
});
test('future ribbon splits exactly at scene boundary using camera capture timestamp',()=>{
 const sections=[{time_s:0,left:[0,0],right:[2,0],connect:false},{time_s:2,left:[0,20],right:[2,20],connect:true}];
 const pieces=D.ribbon(scenes,109,sections);assert.equal(pieces.length,2);
 assert.equal(pieces[0].decision,'turn_right');assert.equal(pieces[1].decision,'keep_lane');
 assert.deepEqual(pieces[0].to.left,[0,10]);assert.deepEqual(pieces[1].from.left,[0,10]);
 assert.equal(D.ribbon(scenes,109,[sections[0],{...sections[1],connect:false}]).length,0);
});
test('empty/unclassified steps do not invent probabilities or select a distant raw sample',()=>{
 assert.equal(D.view([],[],100).lateral,null);
 assert.deepEqual(D.view(scenes,steps,119).rawLateral,[]);
 assert.equal(D.view(scenes,[{t_ns:102e9}],102).rawLongitudinal.length,0);
});
