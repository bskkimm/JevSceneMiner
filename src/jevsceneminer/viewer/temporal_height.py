"""A deterministic height reference shared by every camera frame in a log.

Stationary objects link neighboring times. Build that reference once, smooth
it in time, and query both capture and future times from the same field.
Weak spatial evidence blends toward observed translation rather than deleting
ribbon samples or switching independently fitted references between images.
"""
import numpy as np
from jevsceneminer.viewer.height_reference import HeightReference


class TemporalHeightReference:
    PERIOD=.5
    NOISE_M=.025  # Annotation noise after local temporal averaging.
    MAX_SPAN=22

    def __init__(self,tracks,origin,time_bounds=None):
        self.origin=np.asarray(origin,dtype=float)
        self.available=False
        tracks={k:np.asarray(v,dtype=float) for k,v in tracks.items() if len(v)>=3}
        if not tracks:
            return
        first=min(v[0,0] for v in tracks.values())
        last=max(v[-1,0] for v in tracks.values())
        if time_bounds is not None:
            first=min(first,time_bounds[0]);last=max(last,time_bounds[1])
        self.grid=first+np.arange(max(2,int(np.ceil((last-first)/self.PERIOD))+1))*self.PERIOD
        observations=[{} for _ in self.grid]
        for key,rows in tracks.items():
            lo=max(0,int(np.ceil((rows[0,0]-first-.12)/self.PERIOD)))
            hi=min(len(self.grid),int(np.floor((rows[-1,0]-first+.12)/self.PERIOD))+1)
            for i in range(lo,hi):
                t=self.grid[i];row=HeightReference._nearest(rows,t)
                if row is not None:
                    observations[i][key]=(row,self._bottom(rows,t,row))
        increments=[];translations=[];edges=[];kinds=[];centers=[]
        for i in range(1,len(self.grid)):
            before,after=observations[i-1],observations[i]
            keys=[];xy=[];dz=[]
            for key,(a,za) in before.items():
                if key not in after:
                    continue
                b,zb=after[key]
                if np.linalg.norm(b[1:3]-a[1:3])>.75 or abs(b[6]-a[6])>.15:
                    continue
                if np.any(abs(b[4:6]-a[4:6])>np.maximum(.15,.12*a[4:6])):
                    continue
                keys.append(key);xy.append(a[1:3]-self.origin);dz.append(zb-za)
            xy=np.asarray(xy);dz=np.asarray(dz)
            center=xy.mean(axis=0) if len(xy) else np.full(2,np.nan)
            centers.append(center)
            fit=HeightReference._fit(xy-center,dz) if len(dz)>=4 else None
            if fit is not None:
                coefficient=fit[0].copy()
                local=np.column_stack([(xy-center)/20,np.ones(len(xy))])
                inside=abs(local@coefficient-dz)<=.2
                transform=np.eye(3);transform[2,:2]=-center/20
                influence=transform@np.linalg.pinv(local[inside])
                residual=local[inside]@coefficient-dz[inside]
                model_noise=max(0,float(np.mean(residual**2))-2*self.NOISE_M**2)
                process=model_noise*(influence@influence.T)
                coefficient=transform@coefficient
                increments.append(coefficient);kinds.append(2)
                edges.append((list(np.asarray(keys,dtype=object)[inside]),influence,process))
            elif len(dz)>=2:
                increments.append([0,0,float(np.median(dz))]);kinds.append(1)
                influence=np.zeros((3,len(keys)));influence[2]=1/len(keys)
                transform=np.eye(3);transform[2,:2]=-center/20
                process=transform@np.diag([.03**2,.03**2,0])@transform.T
                edges.append((keys,influence,process))
            else:
                increments.append([np.nan]*3);kinds.append(0)
                edges.append(([],np.empty((3,0)),None))
            translations.append(float(np.median(dz)) if len(dz)>=2 else np.nan)
        increments=np.asarray(increments);translations=np.asarray(translations)
        self.kinds=np.asarray(kinds)
        self.approximate_prefix=np.r_[0,np.cumsum(self.kinds<2)]
        good=np.isfinite(increments[:,0])
        if not good.any():
            return
        # Missing observations are explicitly approximate; continue one shared
        # field instead of reappearing with a different per-frame fallback.
        indices=np.arange(len(increments))
        for j in range(3):
            increments[:,j]=np.interp(indices,indices[good],increments[good,j])
        valid=np.isfinite(translations)
        translations=np.interp(indices,indices[valid],translations[valid])
        centers=np.asarray(centers)
        for j in range(2):
            centers[:,j]=np.interp(indices,indices[good],centers[good,j])
        for i,(keys,influence,process) in enumerate(edges):
            if process is None:
                transform=np.eye(3);transform[2,:2]=-centers[i]/20
                edges[i]=(keys,influence,transform@np.diag([.06**2,.06**2,.1**2])@transform.T)
        self.raw=np.vstack([np.zeros(3),np.cumsum(increments,axis=0)])
        self.coefficients=self._smooth(self.raw)
        self.translation=self._smooth(np.r_[0,np.cumsum(translations)][:,None])[:,0]
        self.covariance=self._covariance(edges)
        self.available=True

    @staticmethod
    def _bottom(rows,t,row):
        a=np.searchsorted(rows[:,0],t-.25);b=np.searchsorted(rows[:,0],t+.25,side='right')
        nearby=rows[a:b]
        if len(nearby)<3:
            return float(row[3]-row[6]/2)
        x=nearby[:,0]-t;y=nearby[:,3]-nearby[:,6]/2
        matrix=np.column_stack([x,np.ones(len(x))])
        weights=np.exp(-.5*(x/.15)**2)
        for _ in range(2):
            root=np.sqrt(weights)
            coefficient=np.linalg.lstsq(matrix*root[:,None],y*root,rcond=None)[0]
            residual=abs(matrix@coefficient-y)
            weights*=np.minimum(1,.08/np.maximum(.08,residual))
        return float(coefficient[1])

    @staticmethod
    def _smooth(values):
        x=np.arange(-3,4);kernel=np.exp(-.5*(x/1.2)**2);kernel/=kernel.sum()
        # Linear edge extension preserves real slopes at log boundaries.
        result=[]
        for column in values.T:
            slope0=(column[min(3,len(column)-1)]-column[0])/min(3,len(column)-1)
            slope1=(column[-1]-column[max(0,len(column)-4)])/min(3,len(column)-1)
            padded=np.r_[column[0]+np.arange(-3,0)*slope0,column,column[-1]+np.arange(1,4)*slope1]
            result.append(np.convolve(padded,kernel,mode='valid'))
        return np.asarray(result).T

    def _covariance(self,edges):
        count=len(self.grid)
        output=np.zeros((count,self.MAX_SPAN+1,3,3))
        for start in range(count):
            weights={};total=np.zeros((3,3))
            for end in range(start+1,min(count,start+self.MAX_SPAN+1)):
                keys,influence,process=edges[end-1]
                total+=process
                for k,key in enumerate(keys):
                    change=influence[:,k]*self.NOISE_M
                    for node,sign in [(end-1,-1),(end,1)]:
                        identity=(node,key);old=weights.get(identity,np.zeros(3));new=old+sign*change
                        # Reused observations cancel between adjacent links.
                        # Summing independent edge variances would overcount
                        # them and prematurely shorten a well-supported path.
                        total+=np.outer(new,new)-np.outer(old,old)
                        weights[identity]=new
                output[start,end-start]=(total+total.T)/2
        return output

    def _sample(self,values,times):
        if values.ndim==1:
            return np.interp(times,self.grid,values)
        return np.column_stack([np.interp(times,self.grid,values[:,j]) for j in range(values.shape[1])])

    def _variance(self,capture_time,times,basis):
        s=np.clip((capture_time-self.grid[0])/self.PERIOD,0,len(self.grid)-1)
        u=np.clip((times-self.grid[0])/self.PERIOD,0,len(self.grid)-1)
        left=int(np.floor(s));right=np.floor(u).astype(int);a=s-left;b=u-right
        covariance=np.zeros((len(times),3,3))
        for i,weight_i in [(left,1-a),(min(left+1,len(self.grid)-1),a)]:
            for j,weight_j in [(right,1-b),(np.minimum(right+1,len(self.grid)-1),b)]:
                start=np.minimum(i,j);span=np.minimum(abs(j-i),self.MAX_SPAN)
                covariance+=(weight_i*weight_j)[:,None,None]*self.covariance[start,span]
        # Variance of an interpolated difference also subtracts the two
        # within-endpoint variances. In particular, identical fractional
        # times must have zero difference and zero uncertainty.
        covariance-=a*(1-a)*self.covariance[left,1]
        covariance-=(b*(1-b))[:,None,None]*self.covariance[right,1]
        return np.maximum(0,np.einsum('ni,nij,nj->n',basis,covariance,basis))

    def offsets(self,capture_time,capture_pose,future_poses):
        missing=dict(available=False,method='shared stationary-landmark height reference',reason='No usable stationary-landmark height timeline')
        poses=np.asarray(future_poses,dtype=float)
        if not self.available or not len(poses):
            return None,missing
        times=poses[:,0]
        basis=np.column_stack([(poses[:,1:3]-self.origin)/20,np.ones(len(poses))])
        coefficients=self._sample(self.coefficients,times)-self._sample(self.coefficients,np.array([capture_time]))[0]
        raw=self._sample(self.raw,times)-self._sample(self.raw,np.array([capture_time]))[0]
        full=np.sum(coefficients*basis,axis=1)
        translation=self._sample(self.translation,times)-self._sample(self.translation,np.array([capture_time]))[0]
        variance=self._variance(capture_time,times,basis)
        variance+=np.sum((coefficients-raw)*basis,axis=1)**2
        uncertainty=np.sqrt(variance)
        blend=np.clip((uncertainty-.25)/.35,0,1)
        weight=1-blend**2*(3-2*blend)
        offsets=np.clip(weight*full+(1-weight)*translation,-5,5)
        first=int(np.clip(np.floor((capture_time-self.grid[0])/self.PERIOD+1e-6),0,len(self.kinds)))
        last=np.clip(np.ceil((times-self.grid[0])/self.PERIOD-1e-6).astype(int),0,len(self.kinds))
        missing_link=(self.approximate_prefix[last]-self.approximate_prefix[first])>0
        approximate=(weight<.999)|missing_link
        info=dict(available=True,method='shared stationary-landmark height reference',
                  supported_fraction=float((~approximate).mean()),usable_fraction=1.0,
                  approximate_fraction=float(approximate.mean()),
                  max_abs_correction_m=float(np.max(abs(offsets))),
                  max_prediction_std_m=float(np.max(uncertainty)))
        return offsets,info
