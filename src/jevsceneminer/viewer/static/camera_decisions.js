/* Camera result timing, used by both the real run and browser-only mock preview. */
(function(root) {
  'use strict';
  function at(scenes,t) {
    const scene=scenes.find(s=>s.start<=t && t<s.end);
    if(!scene)return null;
    let start=scene.start;
    for(const phase of scene.phases) {
      if(start<=t && t<phase.end)return {scene,phase,phaseStart:start};
      start=phase.end;
    }
    return {scene,phase:null,phaseStart:null};
  }
  function ranked(probs) {
    return Object.entries(probs||{}).filter(([,v])=>Number.isFinite(v) && v>=0)
      .sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0]));
  }
  function support(steps,key,label,start,end) {
    const values=steps.filter(r=>start<=r.t_ns/1e9 && r.t_ns/1e9<end)
      .map(r=>r[key]?.[label]).filter(Number.isFinite);
    return values.length ? values.reduce((a,b)=>a+b,0)/values.length : null;
  }
  function view(scenes,steps,t) {
    const active=at(scenes,t);
    let raw=null,delta=Infinity;
    for(const r of steps) {const d=Math.abs(r.t_ns/1e9-t);if(d<delta){delta=d;raw=r;}}
    if(delta>1)raw=null;
    return {rawLateral:ranked(raw?.lateral_probs),rawLongitudinal:ranked(raw?.longitudinal_probs),
      lateral:active?.scene.lateral||null,longitudinal:active?.phase?.decision||null,
      lateralSupport:active?support(steps,'lateral_probs',active.scene.lateral,active.scene.start,active.scene.end):null,
      longitudinalSupport:active?.phase?support(steps,'longitudinal_probs',active.phase.decision,active.phaseStart,active.phase.end):null};
  }
  // Split each quad at exact future scene boundaries. Invalid lens sections stay disconnected.
  function ribbon(scenes,captureTime,sections) {
    const pieces=[];
    for(let i=1;i<sections.length;i++) {
      const a=sections[i-1],b=sections[i];if(!b.connect || b.time_s<=a.time_s)continue;
      const start=captureTime+a.time_s,end=captureTime+b.time_s;
      const cuts=[start,...new Set(scenes.flatMap(s=>[s.start,s.end]).filter(t=>start<t && t<end)),end].sort((x,y)=>x-y);
      const interpolate=t=>{
        const f=(t-start)/(end-start),point=edge=>a[edge].map((x,j)=>x+(b[edge][j]-x)*f);
        return {left:point('left'),right:point('right')};
      };
      for(let j=1;j<cuts.length;j++)pieces.push({from:interpolate(cuts[j-1]),to:interpolate(cuts[j]),
        decision:at(scenes,(cuts[j-1]+cuts[j])/2)?.scene.lateral||null});
    }
    return pieces;
  }
  function mock(start,end,lateralLabels,longitudinalLabels) {
    const scenes=[],steps=[];
    for(let base=start;base<end;base+=40) {
      for(const [a,b,lateral,phases] of [
        [0,14,'keep_lane',[[8,'cruising'],[14,'decelerating']]],
        [14,24,'lane_change_right',[[19,'cruising'],[24,'accelerating']]],
        [24,36,'turn_right',[[28,'decelerating'],[30,'cruising'],[36,'accelerating']]],
        [36,40,'keep_lane',[[40,'cruising']]]]) {
        if(base+a>=end)continue;
        scenes.push({start:base+a,end:Math.min(base+b,end),lateral,
          phases:phases.filter(([p],i)=>i===0 || base+phases[i-1][0]<end)
            .map(([p,decision])=>({end:Math.min(base+p,end),decision}))});
      }
    }
    const distribution=(labels,winner,other)=>{
      const rest=labels.filter(l=>l!==winner && l!==other),probs={[winner]:.78,[other]:.16};
      rest.forEach(l=>probs[l]=.06/rest.length);return probs;
    };
    for(let t=start;t<end;t+=.5) {
      const active=at(scenes,t);if(!active)continue;
      // Demonstrate a raw/merged disagreement without pretending it is another probability.
      const disagree=((t-start)%40)>=25 && ((t-start)%40)<26;
      const winner=disagree?'keep_lane':active.scene.lateral;
      const lateral_probs=distribution(lateralLabels,winner,winner==='keep_lane'?'turn_right':'keep_lane');
      const lon=active.phase.decision;
      const longitudinal_probs=distribution(longitudinalLabels,lon,lon==='cruising'?'accelerating':'cruising');
      steps.push({t_ns:Math.round(t*1e9),lateral:winner,longitudinal:lon,lateral_probs,longitudinal_probs});
    }
    return {scenes,steps};
  }
  const api={at,view,ribbon,mock};
  if(typeof module!=='undefined' && module.exports)module.exports=api;else root.CameraDecisions=api;
})(typeof globalThis!=='undefined'?globalThis:this);
