/* GT scene edits. All operations return fresh scenes so Undo can retain the old draft. */
(function (root) {
  "use strict";
  const EPS = 1e-5, MIN = 0.1;
  const copy = scenes => scenes.map(s => ({...s, ...(s.phases ? {phases: s.phases.map(p => ({...p}))} : {})}));
  function resize(s, start, end) {
    if (s.phases) {
      const kept = []; let a = s.start;
      for (const p of s.phases) { if (p.end > start && a < end) kept.push({end: Math.min(p.end, end), decision: p.decision}); a = p.end; }
      if (!kept.length) kept.push({end, decision: (start >= s.end ? s.phases.at(-1) : s.phases[0]).decision});
      kept[kept.length - 1].end = end; s.phases = kept;
    }
    s.start = start; s.end = end; return s;
  }
  function fromDocument(doc) {
    const out = [];
    for (const raw of [...((doc && doc.scenes) || [])].sort((a,b)=>Number(a.start_ns)-Number(b.start_ns))) {
      const s = {start: Number(raw.start_ns)/1e9, end: Number(raw.end_ns)/1e9,
        lateral: raw.driving_decision || raw.lateral || 'keep_lane',
        phases: raw.longitudinal_phases ? raw.longitudinal_phases.map(p => ({end: Number(p.end_ns)/1e9, decision: p.decision})) : [{end: Number(raw.end_ns)/1e9, decision: raw.longitudinal || 'unknown'}]};
      const prev = out.at(-1);
      if (!raw.longitudinal_phases && prev && prev.lateral === s.lateral && touches(prev.end, s.start)) { prev.end = s.end; prev.phases.push(...s.phases); }
      else out.push(s);
    }
    return out;
  }
  function phaseUpdate(scenes, index, phase, patch) {
    const out = copy(scenes), s = out[index], p = s && s.phases[phase]; if (!p) return out;
    if (patch.end !== undefined && phase < s.phases.length - 1) {
      finite(patch.end); const lo = phase ? s.phases[phase-1].end : s.start, hi = s.phases[phase+1].end;
      if (patch.end <= lo || patch.end >= hi) throw new Error('Phase ends must strictly increase.'); p.end = patch.end;
    }
    if (patch.decision !== undefined) p.decision = patch.decision; return out;
  }
  function phaseSplit(scenes, index, time) {
    const out = copy(scenes), s = out[index]; finite(time); if (!s) return out;
    let a = s.start;
    for (let i=0;i<s.phases.length;i++) { const p=s.phases[i]; if(time>a && time<p.end) { s.phases.splice(i,1,{end:time,decision:p.decision},{...p}); break; } a=p.end; } return out;
  }
  function phaseRemove(scenes,index,phase) {
    const out=copy(scenes),s=out[index]; if(!s || s.phases.length===1 || !s.phases[phase]) return out;
    const end=s.phases[phase].end; s.phases.splice(phase,1); if(phase===s.phases.length) s.phases[phase-1].end=end; return out;
  }
  const touches = (a, b) => Math.abs(a - b) < EPS;
  const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
  function finite(...values) {
    if (!values.every(Number.isFinite)) throw new Error("Scene times must be finite numbers.");
  }
  function update(scenes, index, patch, bounds) {
    const out = copy(scenes), s = out[index];
    if (!s) return out;
    const prev = out[index - 1], next = out[index + 1];
    const old = copy(out);
    const sharedStart = prev && touches(prev.end, s.start);
    const sharedEnd = next && touches(s.end, next.start);
    const start = patch.start ?? s.start, end = patch.end ?? s.end;
    finite(start, end);
    if (patch.start !== undefined && patch.end !== undefined && end - start < MIN - EPS)
      throw new Error("End time must be at least 0.1 seconds after start time.");
    const lo = Math.max(bounds[0], prev ? (sharedStart ? prev.start + MIN : prev.end) : bounds[0]);
    const hi = Math.min(bounds[1], next ? (sharedEnd ? next.end - MIN : next.start) : bounds[1]);
    const requestedEnd = clamp(end, lo + MIN, hi);
    if (patch.start !== undefined) s.start = clamp(start, lo, requestedEnd - MIN);
    s.end = clamp(end, s.start + MIN, hi);
    if (sharedStart) prev.end = s.start;
    if (sharedEnd) next.start = s.end;
    if (s.phases) s.phases = resize(old[index], s.start, s.end).phases;
    if (sharedStart) resize(prev, prev.start, s.start);
    if (sharedEnd && next.phases) next.phases = resize(old[index+1], s.end, next.end).phases;
    if (patch.lateral !== undefined) s.lateral = patch.lateral;
    if (patch.longitudinal !== undefined) s.longitudinal = patch.longitudinal;
    return out;
  }
  function paint(scenes, start, end, labels, bounds) {
    finite(start, end);
    start = clamp(start, bounds[0], bounds[1]); end = clamp(end, bounds[0], bounds[1]);
    if (end - start < MIN - EPS) return copy(scenes);
    return paintInterval(scenes, start, end, labels);
  }
  function paintInterval(scenes, start, end, labels) {
    const out = [];
    for (const s of scenes) {
      if (s.end <= start || s.start >= end) { out.push(copy([s])[0]); continue; }
      if (s.start < start) out.push(resize(copy([s])[0], s.start, start));
      if (s.end > end) out.push(resize(copy([s])[0], end, s.end));
    }
    out.push(labels.phases ? {start,end,lateral:labels.lateral,phases:labels.phases.map(p=>({...p}))} : {start, end, lateral: labels.lateral, longitudinal: labels.longitudinal});
    return out.sort((a, b) => a.start - b.start);
  }
  function paintRow(scenes, start, end, patch, bounds) {
    finite(start, end);
    start = clamp(start, bounds[0], bounds[1]); end = clamp(end, bounds[0], bounds[1]);
    if (end - start < MIN - EPS) return copy(scenes);
    if (scenes.some(s=>s.phases)) {
      if (patch.lateral !== undefined) {
        const phases=[]; let cursor=start;
        for(const s of scenes) {if(s.end<=start || s.start>=end) continue;
          if(s.start>cursor) phases.push({end:Math.min(s.start,end),decision:'unknown'});
          phases.push(...resize(copy([s])[0],Math.max(start,s.start),Math.min(end,s.end)).phases); cursor=Math.min(end,s.end);
        }
        if(cursor<end) phases.push({end,decision:'unknown'});
        return paintInterval(scenes,start,end,{lateral:patch.lateral,phases});
      }
      return copy(scenes).map(s=>{if(s.end<=start || s.start>=end) return s;
        const phases=[];let a=s.start;
        for(const p of s.phases){if(a<start && p.end>start) phases.push({end:start,decision:p.decision});
          if(p.end>start && a<end) phases.push({end:Math.min(p.end,end),decision:patch.longitudinal});
          if(p.end>end || p.end<=start) phases.push({...p}); a=p.end;}
        s.phases=phases;return s;});
    }
    const cuts = new Set([start, end]);
    for (const s of scenes) {
      if (s.start > start && s.start < end) cuts.add(s.start);
      if (s.end > start && s.end < end) cuts.add(s.end);
    }
    const times = [...cuts].sort((a, b) => a - b);
    let out = copy(scenes);
    for (let i = 0; i + 1 < times.length; i++) {
      const a = times[i], b = times[i + 1], mid = (a + b) / 2;
      const existing = scenes.find(s => s.start <= mid && mid < s.end);
      const labels = {lateral: existing ? existing.lateral : "keep_lane",
                      longitudinal: existing ? existing.longitudinal : "cruising", ...(existing && existing.phases ? {phases: resize(copy([existing])[0],a,b).phases} : {}), ...patch};
      out = paintInterval(out, a, b, labels);
    }
    return out;
  }
  function split(scenes, index, time) {
    finite(time);
    const out = copy(scenes), s = out[index];
    if (!s || time - s.start < MIN - EPS || s.end - time < MIN - EPS) return out;
    out.splice(index, 1, resize(copy([s])[0],s.start,time), resize(copy([s])[0],time,s.end));
    return out;
  }
  function mergeLeft(scenes, index) {
    const out = copy(scenes), prev = out[index - 1], s = out[index];
    if (!prev || !s || !touches(prev.end, s.start)) return out;
    if(prev.phases && s.phases) prev.phases.push(...s.phases); prev.end = s.end; out.splice(index, 1); return out;
  }
  function remove(scenes, index) {
    const out = copy(scenes), s = out[index];
    if (!s) return out;
    const prev = out[index - 1], next = out[index + 1];
    if (prev && touches(prev.end, s.start)) resize(prev,prev.start,s.end);
    else if (next && touches(s.end, next.start)) resize(next,s.start,next.end);
    out.splice(index, 1); return out;
  }
  const api = {update, paint, paintRow, split, mergeLeft, remove, copy, fromDocument, phaseUpdate, phaseSplit, phaseRemove};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SceneEditor = api;
})(globalThis);
