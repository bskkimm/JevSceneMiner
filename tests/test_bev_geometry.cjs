const {test}=require('node:test');
const assert=require('node:assert/strict');
const B=require('../src/jevsceneminer/viewer/static/bev_geometry.js');
test('native car footprint uses the recorded dimensions',()=>{
 assert.deepEqual(B.objectFootprint(['car',2,3,0,0,5.95,2.16],[4.6,1.9]),{x:2,y:3,yaw:0,length:5.95,width:2.16});
 assert.equal(B.objectFootprint(['car',2,3,0,0],[4.6,1.9]).length,4.6);
});
test('nuPlan ego footprint center is ahead of the recorded rear axle',()=>{
 const v={length_m:5.176,width_m:2.297,rear_axle_to_center_m:1.461};
 const p=B.egoFootprint([0,10,20,Math.PI/2],v);
 assert.ok(Math.abs(p.x-10)<1e-10);assert.equal(p.y,21.461);
 assert.equal(p.length,5.176);assert.equal(p.width,2.297);
});

test('fitting shows the entire 80m future while the old viewport clips it',()=>{
 const W=600,H=450,pose=[0,0,0];
 const sections=[{left:[0,1],right:[0,-1]},{left:[80,1],right:[80,-1]}];
 const oldScale=W/90;
 assert.ok(H*.65-80*oldScale<0);
 const scale=B.fitScale(pose,sections,W,H,oldScale);
 for(const section of sections)for(const point of [section.left,section.right]) {
  const [x,y]=B.screenPoint(point,pose,W,H,scale);
  assert.ok(x>=0&&x<=W&&y>=0&&y<=H,`clipped ${x},${y}`);
 }
 assert.ok(scale<oldScale);
});
test('viewport fitting handles lateral bends, behind-ego paths, and short paths',()=>{
 const W=600,H=450,pose=[100,200,Math.PI/2],base=W/90;
 const sections=[{left:[100,200],right:[101,200]},{left:[180,150],right:[181,150]}];
 const scale=B.fitScale(pose,sections,W,H,base);
 for(const s of sections)for(const p of [s.left,s.right]) {
  const [x,y]=B.screenPoint(p,pose,W,H,scale);
  assert.ok(x>=0&&x<=W&&y>=0&&y<=H);
 }
 assert.equal(B.fitScale(pose,[],W,H,base),base);
 assert.equal(B.fitScale(pose,[{left:[100,210],right:[101,210]}],W,H,base),base);
});

test('fitting keeps trajectory clear of labels reserved at the top',()=>{
 const W=600,H=450,pose=[0,0,0],sections=[{left:[0,1],right:[0,-1]},{left:[80,1],right:[80,-1]}];
 const options={anchor:.85,topPadding:120};
 const scale=B.fitScale(pose,sections,W,H,W/90,options);
 for(const section of sections)for(const point of [section.left,section.right]) {
  const [x,y]=B.screenPoint(point,pose,W,H,scale,options.anchor);
  assert.ok(y>=120&&y<=H,`point ${y} hidden by top labels`);
 }
});
