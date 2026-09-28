import os, numpy as np, pandas as pd, importlib.util
spec=importlib.util.spec_from_file_location('fv','temp/GLM53/scripts/f_varying_load.py'); fv=importlib.util.module_from_spec(spec); spec.loader.exec_module(fv)

class CppMedianV3(fv.CompV3):
    def _upper_median(self, a):
        a=np.sort(a)
        return float(a[len(a)//2])
    def process(self, ts, v):
        n=len(v)
        if n<=0: return
        if n!=self.n:
            self._reset_for(n); self.fast=self.slow=0.0
        p=self.p; total=float(np.sum(v))
        if not self.init:
            self.init=True; self.last_ts=ts; self.t0=ts; self.ts=self.fast=self.slow=total; self.min_ts=self.max_ts=total; dt=0.0
        else:
            dt=min(max(ts-self.last_ts,0.0),0.1); self.last_ts=ts
            if dt>0:
                self.ts += (dt/p['tau_total'])*(total-self.ts)
                self.fast += (dt/p['tau_fast'])*(total-self.fast)
                self.slow += (dt/p['tau_slow'])*(total-self.slow)
        self.min_ts=min(self.min_ts,self.ts); self.max_ts=max(self.max_ts,self.ts)
        if dt>0: self.level += (dt/p['tau_level'])*(self.ts-self.level)
        eps=1e-6*(1.0+abs(self.max_ts))
        div=abs(self.fast-self.slow); div_onset_thr=p['onset_rel']*max(self.slow,eps)
        div_step_thr=max(p['step_rel']*max(self.slow,eps), p['step_abs_frac']*self.max_ts)
        u=(ts-self.onset) if self.in_load else 0.0
        if self.in_load: thr=div_step_thr; step_now=div>thr
        else: thr=max(div_onset_thr,p['step_abs_frac']*self.max_ts); step_now=div>thr and self.fast>self.slow
        if step_now:
            if self.pend_t is None: self.pend_t=ts
        elif self.pend_t is not None and div < 0.5*thr: self.pend_t=None
        step_confirmed=self.pend_t is not None and ts-self.pend_t>p['step_persist']
        def is_idle(): return self.ts < p['idle_frac']*max(self.level,eps) or self.ts < 1.5*self.min_ts+eps
        if not self.in_load:
            if self.t0 is None or ts <= self.t0+1e-9:
                if total>eps: self._onset(ts,'onset'); self._snap(self.ts)
            elif step_confirmed: self._onset(ts,'onset'); self._snap(self.fast)
            self.hold=False
        else:
            if u>p['unload_fast'] and is_idle():
                self.in_load=False; self.armed=True; self._snap(self.ts); self.hold=False; self.hold_comp=None; self.pend_t=None; self.events.append((ts,'unload'))
            elif u>p['step_suppress'] and step_confirmed:
                if is_idle():
                    self.in_load=False; self.armed=True; self._snap(self.ts); self.hold=False; self.hold_comp=None; self.pend_t=None; self.events.append((ts,'unload'))
                else:
                    Z=v-(self.b if self.b is not None else 0.0)
                    self.in_load=True; self.onset=ts; self.a_acc=np.zeros(n); self.a_frames=0
                    self.A=(self.a_new_acc/max(self.a_new_frames,1) if self.a_new_frames else Z.copy())
                    amax=float(np.max(self.A)); self.loaded=(self.A>p['loaded_frac']*amax if amax>1e-9 else np.zeros(n,bool)); self.a_captured=True
                    ld=self.loaded & (self.A>1e-9)
                    if np.any(ld) and self.hold_comp is not None:
                        ratio=self.hold_comp[ld]/np.maximum(self.gamma[ld]*self.A[ld],1e-9)
                        self.g=float(np.clip(self._upper_median(ratio),0.0,1.0))
                    else: self.g=0.0
                    self.hold=False; self.hold_comp=None; self.a_new_acc=np.zeros(n); self.a_new_frames=0; self.pend_t=None; self._snap(self.fast); self.events.append((ts,'restep'))
            elif self.pend_t is not None:
                self.hold=True
                if self.hold_comp is None and self.a_captured and self.b is not None:
                    ld0=self.loaded & (self.A>1e-9); comp0=np.zeros(n)
                    comp0[ld0]=np.clip(self.gamma[ld0]*self.A[ld0]*self.g,p['creep_lo']*self.A[ld0],p['creep_hi']*self.A[ld0]); self.hold_comp=comp0
                if self.b is not None: self.a_new_acc += v-self.b; self.a_new_frames+=1
            else:
                self.hold=False; self.hold_comp=None; self.a_new_acc=np.zeros(n); self.a_new_frames=0
        if not self.in_load and self.armed and self.ts < p['base_gate_frac']*self.max_ts: self.b += (dt/p['tau_base'])*(v-self.b)
        Z=v-self.b
        if not self.in_load: return Z
        if not self.a_captured:
            if p['a_w0']<=u<=p['a_w1']: self.a_acc+=Z; self.a_frames+=1
            if u>p['a_w1']:
                self.A=self.a_acc/max(self.a_frames,1); amax=float(np.max(self.A)); self.loaded=self.A>p['loaded_frac']*amax if amax>1e-9 else np.zeros(n,bool); self.a_captured=True
            else: return Z
        ld=self.loaded & (self.A>1e-9)
        if self.hold:
            if self.hold_comp is None:
                comp0=np.zeros(n); comp0[ld]=np.clip(self.gamma[ld]*self.A[ld]*self.g,p['creep_lo']*self.A[ld],p['creep_hi']*self.A[ld]); self.hold_comp=comp0
            out=Z.copy(); out[ld]-=self.hold_comp[ld]; return out
        rel=(Z[ld]-self.A[ld])/self.A[ld]
        if rel.size: self.g += (dt/p['tau_creep'])*(self._upper_median(rel)-self.g)
        if self.g>p['g_enable']:
            self.g2+=dt*self.g*self.g; full=np.zeros(n); full[ld]=rel; self.g_rel+=dt*self.g*full
            if self.g2>1e-8:
                g_new=np.ones(n); g_new[ld]=np.clip(self.g_rel[ld]/self.g2,p['gamma_lo'],p['gamma_hi']); self.gamma=g_new
        out=Z.copy(); creep=np.zeros(n); creep[ld]=np.clip(self.gamma[ld]*self.A[ld]*self.g,p['creep_lo']*self.A[ld],p['creep_hi']*self.A[ld]); out[ld]-=creep[ld]
        return out

for tag,(base,loc) in [('A',('变化负载','零负载-切换负载-零负载-再切换负载')),('B',('变化负载','零负载-中途切换负载-零负载-切换负载'))]:
    t,X,tc=fv.load_csv(os.path.join('temp',base,loc,'device_001_seg000.csv'))
    c=CppMedianV3(); Y=np.empty_like(X,float)
    for i in range(len(t)): Y[i]=c.process(t[i],X[i].astype(float))
    cpp=pd.read_csv(f'temp/glm53_cpp_harness/cpp_{tag}.csv',header=None,names=['i','t','tot'])
    d=cpp['tot'].to_numpy()-Y.sum(axis=1)
    print(tag,'max_abs',np.max(np.abs(d)),'rms',np.sqrt(np.mean(d*d)),'argmax',np.argmax(np.abs(d)), 't', cpp['t'].iloc[np.argmax(np.abs(d))])
