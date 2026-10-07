// Coordinates of nuPlan boxes: detections use their center; ego uses its rear axle.
(function(root) {
  "use strict";
  const positive=(v,fallback)=>Number.isFinite(v)&&v>0?v:fallback;
  function objectFootprint(row,fallback) {
    return {x:row[1],y:row[2],yaw:row[3],length:positive(row[5],fallback[0]),width:positive(row[6],fallback[1])};
  }
  function egoFootprint(track,vehicle) {
    const [,x,y,yaw]=track, d=vehicle.rear_axle_to_center_m;
    return {x:x+d*Math.cos(yaw),y:y+d*Math.sin(yaw),yaw,length:vehicle.length_m,width:vehicle.width_m};
  }
  function screenPoint(point,pose,W,H,scale,anchor=.65) {
    const dx=point[0]-pose[0],dy=point[1]-pose[1],c=Math.cos(pose[2]),s=Math.sin(pose[2]);
    return [W/2+(s*dx-c*dy)*scale,H*anchor-(c*dx+s*dy)*scale];
  }
  function fitScale(pose,sections,W,H,baseScale,options={}) {
    let scale=baseScale;
    const anchor=options.anchor ?? .65,top=options.topPadding || 0;
    const c=Math.cos(pose[2]),s=Math.sin(pose[2]),margin=6;
    for(const section of sections)for(const point of [section.left,section.right]) {
      const dx=point[0]-pose[0],dy=point[1]-pose[1];
      const f=c*dx+s*dy,r=s*dx-c*dy;
      if(f>0)scale=Math.min(scale,Math.max(1,H*anchor-top)/(f+margin));
      if(f<0)scale=Math.min(scale,H*(1-anchor)/(-f+margin));
      scale=Math.min(scale,W*.5/(Math.abs(r)+margin));
    }
    return scale;
  }
  const api={objectFootprint,egoFootprint,screenPoint,fitScale};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  else root.BevGeometry=api;
})(typeof globalThis!=="undefined"?globalThis:this);
