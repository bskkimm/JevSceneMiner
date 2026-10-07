"""Align future heights to the capture frame using stationary landmarks.

The log's vertical datum and tilt can move over time even for stationary
actors. Fit their observed height changes as a plane over XY, then subtract
that frame change at each future ego position. This preserves residual road
grade and never adjusts logged XY, camera calibration or source poses.
"""
from itertools import combinations
import numpy as np


class HeightReference:
    def __init__(self, tracks):
        self.tracks={key:np.asarray(rows,dtype=float) for key,rows in tracks.items() if len(rows)>=3}

    @staticmethod
    def _nearest(rows,t):
        k=int(np.searchsorted(rows[:,0],t))
        k=min(range(max(0,k-1),min(len(rows),k+1)),key=lambda i:abs(rows[i,0]-t))
        return rows[k] if abs(rows[k,0]-t)<=.12 else None

    @staticmethod
    def _fit(xy,dz):
        if len(dz)<4:
            return None
        A=np.column_stack([xy/20,np.ones(len(xy))])
        # A line of landmarks cannot constrain a two-dimensional tilt.
        if np.linalg.svd(xy-xy.mean(axis=0),compute_uv=False)[-1]<1:
            return None
        choices=list(combinations(range(len(dz)),3))
        if len(choices)>64:
            choices=[choices[i] for i in np.random.default_rng(0).choice(len(choices),64,replace=False)]
        best=None
        for choice in choices:
            matrix=A[list(choice)]
            if np.linalg.cond(matrix)>100:
                continue
            coef=np.linalg.solve(matrix,dz[list(choice)])
            residual=abs(A@coef-dz)
            inside=residual<=.15
            score=(int(inside.sum()),-float(np.median(residual)))
            if best is None or score>best[0]:
                best=(score,inside)
        if best is None or best[0][0]<max(4,int(np.ceil(.6*len(dz)))):
            return None
        inside=best[1]
        if np.linalg.svd(xy[inside]-xy[inside].mean(axis=0),compute_uv=False)[-1]<1:
            return None
        coef=np.linalg.lstsq(A[inside],dz[inside],rcond=None)[0]
        if np.linalg.norm(coef[:2])/20>.08 or np.max(abs(A[inside]@coef-dz[inside]))>.20:
            return None
        # A residual-only gate misses weak geometry: several landmarks in a
        # tight cluster can fit perfectly while amplifying centimetres of
        # annotation noise into metres at a distant future position.
        noise=max(.05,float(np.sqrt(np.mean((A[inside]@coef-dz[inside])**2))))
        covariance=noise**2*np.linalg.inv(A[inside].T@A[inside])
        return coef,covariance

    def offsets(self, capture_time, capture_pose, future_poses):
        future_poses=np.asarray(future_poses,dtype=float)
        origin=np.asarray(capture_pose[:2])
        missing=dict(available=False,method='stationary-landmark height reference',
                     reason='Insufficient stationary landmarks with non-collinear geometry')
        if len(future_poses)==0:
            return None,missing
        end=min(10,float(future_poses[-1,0]-capture_time))
        if end<=0:
            return None,missing
        # Include landmarks that enter view later. Overlapping neighboring
        # observations can transfer the capture reference without extrapolation.
        candidates={key:rows for key,rows in self.tracks.items()
                    if rows[-1,0]>=capture_time-.12 and rows[0,0]<=capture_time+end+.12
                    and np.linalg.norm(rows[max(0,min(len(rows)-1,
                        np.searchsorted(rows[:,0],capture_time))),1:3]-origin)<90}
        def observed(t):
            return {key:row for key,rows in candidates.items()
                    if (row:=self._nearest(rows,t)) is not None}
        current=observed(capture_time)
        grid=np.unique(np.r_[0,np.arange(.5,end,.5),end])
        def difference(before,after):
            xy=[];dz=[]
            for key,a in before.items():
                b=after.get(key)
                if b is None or np.linalg.norm(b[1:3]-a[1:3])>.75 or abs(b[6]-a[6])>.15:
                    continue
                if np.any(abs(b[4:6]-a[4:6])>np.maximum(.15,.12*a[4:6])):
                    continue
                xy.append(a[1:3]-origin)
                dz.append((b[3]-b[6]/2)-(a[3]-a[6]/2))
            return self._fit(np.asarray(xy),np.asarray(dz)) if len(dz)>=4 else None
        coefficients=[np.zeros(3)]
        covariances=[np.zeros((3,3))]
        previous=current
        bridged=0
        for dt in grid[1:]:
            after=observed(capture_time+dt)
            # Re-anchor directly whenever supported; only chain when the
            # original objects have left view. Never bridge an unobserved gap.
            fit=difference(current,after)
            if fit is None and np.isfinite(coefficients[-1]).all():
                change=difference(previous,after)
                if change is not None:
                    fit=(coefficients[-1]+change[0],covariances[-1]+change[1])
                    bridged+=1
            if fit is not None and np.linalg.norm(fit[0][:2])/20>.08:
                fit=None
            coefficients.append(fit[0] if fit is not None else [np.nan]*3)
            covariances.append(fit[1] if fit is not None else np.full((3,3),np.nan))
            previous=after
        coefficients=np.asarray(coefficients)
        covariances=np.asarray(covariances)
        good=np.isfinite(coefficients).all(axis=1)
        if good.sum()<2:
            return None,missing
        smooth=coefficients.copy()
        for i in range(1,len(grid)-1):
            if good[i-1:i+2].all():
                smooth[i]=np.median(coefficients[i-1:i+2],axis=0)
        offsets=np.full(len(future_poses),np.nan)
        relative=future_poses[:,0]-capture_time
        uncertain=np.zeros(len(future_poses),dtype=bool)
        for i in range(1,len(grid)):
            if not good[i-1:i+1].all():
                continue
            chosen=(relative>=grid[i-1]-1e-6)&(relative<=grid[i]+1e-6)
            f=np.clip((relative[chosen]-grid[i-1])/(grid[i]-grid[i-1]),0,1)
            coef=(1-f[:,None])*smooth[i-1]+f[:,None]*smooth[i]
            positions=(future_poses[chosen,1:3]-origin)/20
            basis=np.column_stack([positions,np.ones(len(positions))])
            values=np.sum(coef*basis,axis=1)
            covariance=(1-f[:,None,None])*covariances[i-1]+f[:,None,None]*covariances[i]
            variance=np.einsum('ni,nij,nj->n',basis,covariance,basis)
            original=(1-f[:,None])*coefficients[i-1]+f[:,None]*coefficients[i]
            smoothing_error=np.sum((coef-original)*basis,axis=1)
            #25cm maximum prediction standard error, carrying every bridge's
            # uncertainty forward. Unsupported points break the camera ribbon.
            acceptable=variance+smoothing_error**2<=.25**2
            uncertain[chosen]=~acceptable
            offsets[chosen]=np.where((abs(values)<=5)&acceptable,values,np.nan)
        valid=np.isfinite(offsets)
        info=dict(available=bool(valid.any()),method='stationary-landmark height reference',
                  capture_landmarks=len(current),supported_fraction=float(valid.mean()),
                  max_abs_correction_m=float(np.max(abs(offsets[valid]))) if valid.any() else 0,
                  grid_intervals=int(good.sum()),bridged_grid_nodes=bridged,
                  unsupported_uncertainty_fraction=float(uncertain.mean()))
        return offsets,info
