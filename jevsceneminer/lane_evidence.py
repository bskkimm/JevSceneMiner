"""Body geometry and a continuous, explicitly window-local lane reference.

Map-derived occupancy is evidence, not a maneuver label or painted-line calibration.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely import STRtree

from .facts import DT_S, LANE_EVERY

DEPARTURE_FRACTION = .01  # numerical reporting tolerance, never a label rule


def rectangle(x, y, yaw, length, width, center_forward=0):
    if not all(math.isfinite(v) for v in (x,y,yaw,length,width,center_forward)) or min(length,width)<=0:
        return None
    c,s=math.cos(yaw),math.sin(yaw)
    corners=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*[length/2,width/2]
    return Polygon(corners@np.array([[c,s],[-s,c]])+[x+center_forward*c,y+center_forward*s])


class LaneEvidence:
    def __init__(self, tl, lanemap, lo, hi, i_now):
        self.tl,self.lanemap,self.lo,self.hi,self.i_now=tl,lanemap,lo,hi,i_now
        self.geometry=getattr(tl.session,'ego_geometry',None)
        if self.geometry is not None:
            g=self.geometry
            if (not all(math.isfinite(g.get(k,math.nan)) for k in ('length_m','width_m','center_forward_m'))
                    or min(g['length_m'],g['width_m'])<=0):
                self.geometry=None
        self.polygons=getattr(lanemap,'polygons',{})
        self.indices=sorted({lo,hi,i_now,*range(lo+(-lo)%LANE_EVERY,hi+1,LANE_EVERY)})
        self.reference_ids=[]
        # Anchor to the first observed lane in this window, never silently re-anchor
        # after a neighbor transition or a gap. This is not an inferred maneuver onset.
        previous_i=None
        for i in self.indices:
            p=tl.lane(i) if tl.valid[i] else None
            if p is None:
                continue
            if not self.reference_ids or p.lane_id!=self.reference_ids[-1]:
                if self.reference_ids and lanemap.relation(self.reference_ids[-1],p.lane_id)!='following':break
                self.reference_ids.append(p.lane_id)
            previous_i=i
        self.observed_reference_ids=list(self.reference_ids)
        # Support the full body at window ends with unambiguous map connections.
        # Do not union fork alternatives or neighboring lanes.
        for direction in ("previous","following"):
            covered=0.
            for _ in range(8):
                if not self.reference_ids:break
                edge=self.reference_ids[0 if direction=="previous" else -1]
                ids=[lid for lid in getattr(lanemap.lanes[edge],direction) if lid in lanemap.lanes]
                if len(ids)!=1 or ids[0] in self.reference_ids:break
                lid=ids[0]
                if direction=="previous":self.reference_ids.insert(0,lid)
                else:self.reference_ids.append(lid)
                covered+=LineString(lanemap.lanes[lid].centerline).length
                if covered>=5:break
        coords=[];join_gaps=[]
        for lid in self.reference_ids:
            pts=np.asarray(lanemap.lanes[lid].centerline)
            if coords:
                gap=float(np.linalg.norm(np.asarray(coords[-1])-pts[0]));join_gaps.append(gap)
                if gap>5:break
                # Blend endpoints rather than insert a sideways connector and reset
                # the offset at that connector. Native polygons remain untouched.
                coords[-1]=((np.asarray(coords[-1])+pts[0])/2).tolist()
                coords.extend(pts[1:].tolist())
            else:coords=pts.tolist()
        self.line=LineString(coords) if len(coords)>1 else None
        shapes=[self.polygons[lid] for lid in self.reference_ids if lid in self.polygons]
        self.reference_area=unary_union(shapes) if shapes and len(shapes)==len(self.reference_ids) else None
        boundary=self.reference_area.boundary if self.reference_area is not None else None
        lines=list(boundary.geoms) if boundary is not None and hasattr(boundary,"geoms") else [boundary] if boundary is not None else []
        self.boundary_edges=[LineString([a,b]) for line in lines for a,b in zip(list(line.coords)[:-1],list(line.coords)[1:]) if a!=b]
        self.boundary_tree=STRtree(self.boundary_edges)
        self.join_gap_m=max(join_gaps,default=0.)
        self._cache={}

    def ego_shape(self, i):
        if self.geometry is None or not self.tl.valid[i]:return None
        g=self.geometry;t=self.tl
        return rectangle(t.x[i],t.y[i],t.yaw[i],g['length_m'],g['width_m'],g['center_forward_m'])

    def reference_frame(self, i):
        if self.line is None or not self.tl.valid[i]:return None
        point=Point(self.tl.x[i],self.tl.y[i]);station=self.line.project(point)
        foot=self.line.interpolate(station)
        a=self.line.interpolate(max(0,station-.05));b=self.line.interpolate(min(self.line.length,station+.05))
        length=math.hypot(b.x-a.x,b.y-a.y)
        if length<1e-9:return None
        tangent=np.array([(b.x-a.x)/length,(b.y-a.y)/length]);normal=np.array([-tangent[1],tangent[0]])
        delta=np.array([point.x-foot.x,point.y-foot.y])
        # Extrapolation beyond a reference end does not establish lane departure.
        if station<1e-6 or station>self.line.length-1e-6:
            if abs(float(delta@tangent))>.1:return None
        offset=math.copysign(point.distance(self.line),float(delta@normal))
        return foot,normal,offset,tangent

    def centered_shape(self,i):
        f=self.reference_frame(i)
        if f is None or self.geometry is None:return None
        foot,_,_,tangent=f;g=self.geometry
        return rectangle(foot.x,foot.y,math.atan2(tangent[1],tangent[0]),g['length_m'],g['width_m'],g['center_forward_m'])

    def at(self,i):
        if i in self._cache:return self._cache[i]
        shape=self.ego_shape(i);frame=self.reference_frame(i);p=self.tl.lane(i) if self.tl.valid[i] else None
        row=dict(time_s=(i-self.i_now)*DT_S,ego_data_available=bool(self.tl.valid[i]),
            matched_lane_id=p.lane_id if p else None,reference_offset_m=frame[2] if frame else None,
            lane_relative_heading_deg=math.degrees(math.atan2(math.sin(self.tl.yaw[i]-math.atan2(frame[3][1],frame[3][0])),
                                      math.cos(self.tl.yaw[i]-math.atan2(frame[3][1],frame[3][0])))) if frame else None,
            reference_lane_fraction=None,outside_reference_fraction=None,left_boundary_margin_m=None,
            right_boundary_margin_m=None,lane_occupancy=[],mapped_fraction=None,unmapped_fraction=None,
            ambiguous_overlap_fraction=None,lateral_departure=None,crossed_boundary_sides=[])
        if shape is not None and self.polygons and p is not None:
            center=shape.centroid;radius=math.hypot(self.geometry['length_m'],self.geometry['width_m'])/2
            ids={lid for _,lid in self.lanemap._find_within(center.x,center.y,radius) if lid in self.polygons}
            hits=[]
            for lid in sorted(ids):
                overlap=shape.intersection(self.polygons[lid])
                if overlap.area>1e-8:
                    angle=abs(self.lanemap.project(lid,self.tl.x[i],self.tl.y[i],self.tl.yaw[i]).heading_diff)
                    direction="opposing" if angle>math.radians(120) else "same-direction" if angle<math.radians(60) else "crossing-direction"
                    hits.append(overlap)
                    row['lane_occupancy'].append(dict(lane_id=lid,fraction=min(1.,max(0.,float(overlap.area/shape.area))),
                        direction_relation=direction,relation=self.lanemap.relation(p.lane_id,lid),kind=self.lanemap.lanes[lid].kind,
                        intersection=self.lanemap.lanes[lid].is_intersection))
            union=unary_union(hits);mapped=min(1.,max(0.,float(union.area/shape.area)))
            row.update(mapped_fraction=mapped,unmapped_fraction=max(0,1-mapped))
            ambiguous=unary_union([a.intersection(b) for k,a in enumerate(hits) for b in hits[k+1:]])
            row['ambiguous_overlap_fraction']=min(1.,max(0.,float(ambiguous.area/shape.area)))
            if self.reference_area is not None and frame is not None:
                inside=min(1.,max(0.,float(shape.intersection(self.reference_area).area/shape.area)))
                row.update(reference_lane_fraction=inside,outside_reference_fraction=max(0,1-inside))
                foot,normal,_,_=frame
                origin=np.array([foot.x,foot.y]);section=LineString([origin-50*normal,origin+50*normal])
                cut=self.reference_area.intersection(section)
                parts=list(cut.geoms) if hasattr(cut,'geoms') else [cut]
                spans=[]
                for part in parts:
                    if part.geom_type!='LineString' or part.is_empty:continue
                    values=(np.asarray(part.coords)-origin)@normal
                    if values.min()<=0<=values.max():spans.append((float(values.min()),float(values.max())))
                if len(spans)==1:
                    low,high=spans[0];body=(np.asarray(shape.exterior.coords)-origin)@normal
                    row.update(left_boundary_margin_m=float(high-body.max()),right_boundary_margin_m=float(body.min()-low))
                # Lateral boundary segments are longitudinally oriented. End caps
                # and map holes must not manufacture lane-departure events.
                sides=[]
                for k in self.boundary_tree.query(shape,predicate="intersects"):
                    edge=self.boundary_edges[k];a,b=np.asarray(edge.coords);v=b-a;length=float(np.linalg.norm(v))
                    if length<1e-8 or abs(float(v@frame[3]))/length<.5:continue
                    if shape.intersection(edge).length>.01:
                        side="left" if float(((a+b)/2-origin)@normal)>0 else "right"
                        sides.append(side)
                row["crossed_boundary_sides"]=sorted(set(sides))
                lateral=bool(sides)
                if outside:=row["outside_reference_fraction"]:
                    if outside>.999 and len(spans)==1:
                        lateral |= bool(body.min()>high or body.max()<low)
                if row["outside_reference_fraction"]>DEPARTURE_FRACTION and not lateral and not spans:
                    row["lateral_departure"]=None  # missing cross-section cannot establish return
                else:
                    row["lateral_departure"]=bool(row["outside_reference_fraction"]>DEPARTURE_FRACTION and lateral)
        self._cache[i]=row
        return row

    def summary(self):
        observations=[self.at(i) for i in self.indices]
        spans=[];current=None;previous=None
        for row in observations:
            known=row['lateral_departure'] is not None
            contiguous=(previous is not None and self.tl.valid[
                self.i_now+round(previous['time_s']/DT_S):self.i_now+round(row['time_s']/DT_S)+1].all()
                and previous['lateral_departure'] is not None)
            if current and (not known or not contiguous):spans.append(current);current=None
            outside=known and row['lateral_departure']
            if outside:
                if current is None:
                    current=dict(first_outside_s=row['time_s'],last_outside_s=row['time_s'],return_s=None,
                        entry_observed=bool(contiguous and not previous['lateral_departure']))
                current['last_outside_s']=row['time_s']
            elif current:
                current['return_s']=row['time_s'];spans.append(current);current=None
            previous=row
        if current:spans.append(current)
        known=[r for r in observations if r['outside_reference_fraction'] is not None]
        offsets=[r for r in observations if r['reference_offset_m'] is not None]
        peak=max(known,key=lambda r:r['outside_reference_fraction']) if known else None
        lateral_peak=max(offsets,key=lambda r:abs(r['reference_offset_m']-offsets[0]['reference_offset_m'])) if offsets else None
        return dict(ego_geometry=self.geometry,reference_lane_ids=self.reference_ids,
            observed_reference_lane_ids=self.observed_reference_ids,
            reference_definition='first observed connected lane chain plus unique end connections for body coverage; not inferred maneuver origin',
            centerline_join_method='midpoint blend of connected endpoints',max_centerline_join_gap_m=self.join_gap_m,
            map_boundary_alignment='unvalidated',overlap_rule='per-segment shares can overlap; mapped total uses union',
            departure_reporting_fraction=DEPARTURE_FRACTION,observations=observations,now=self.at(self.i_now),
            departure_intervals=spans,peak_body_departure=peak,lateral_excursion_peak=lateral_peak,
            from_s=(self.lo-self.i_now)*DT_S,to_s=(self.hi-self.i_now)*DT_S)
