import numpy as np, structure, time
from ldnet_aero import LDNetAero
from controller import Controller
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
a=LDNetAero('models/latent_10');a.reset(dt=DT)
def res(xe):
    cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U)
    return [-q*cl-structure.K_H*xe[0], q*cm*C-structure.K_ALPHA*xe[1]]
xe=fsolve(res,[0.,0.]);x0=np.array([xe[0],0.,xe[1],0.])
cl0,cm0=a.predict(x0,0.,0.,U);CLt=float(cl0);Fyt=q*CLt;Mzt=q*float(cm0)*C
ctrl=Controller(a.predict,U=U,dt=DT,Q_h=0,Q_alpha=0,Q_alpha_dot=3e5,Q_CL=1e3,R=0.3,delta_max=20.,delta_dot_max=100.,C_L_trim=CLt,Fy_trim=Fyt,Mz_trim=Mzt,global_search=True,n_grid=9,causal_basin=True,target_lpf=0.0,mpc_horizon=6,aero=a);ctrl.reset()
# evolve to a mid-gust state (W0=20)
a.reset(dt=DT);x=x0.copy()
for k in range(150):
    W=(20/2)*(1-np.cos(2*np.pi*(k*DT)/1.0)); cl,cm=a.predict(x,0.,W,U);a.advance(x,0.,W,U,DT)
    Fy=q*cl-Fyt;Mz=q*cm*C-Mzt;x=structure.step_rk4(x,Fy,Mz,DT)
W=(20/2)*(1-np.cos(2*np.pi*(150*DT)/1.0))
grid=np.linspace(0.,20.,9)
t0=time.time(); Jb=ctrl._rollout_cost_batch(grid,x,W); tb=time.time()-t0
t0=time.time(); Js=np.array([ctrl._rollout_cost(float(d),x,W) for d in grid]); ts=time.time()-t0
print('max rel diff J = %.2e'%np.max(np.abs(Jb-Js)/(np.abs(Js)+1e-9)))
print('argmin batch=%.2f  slow=%.2f'%(grid[np.argmin(Jb)],grid[np.argmin(Js)]))
print('time: batch=%.3fs  slow=%.3fs  speedup=%.1fx'%(tb,ts,ts/tb))
