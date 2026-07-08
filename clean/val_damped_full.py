import numpy as np, structure, csv, json
from ldnet_aero import LDNetAero
from scipy.optimize import fsolve
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S;QD=q
def gust(t,W0=10.): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
for MD in ['models/latent_10','models_damped/latent_10']:
    lam=json.load(open(MD+'/config.json')).get('lambda_damp',0.0)
    a=LDNetAero(MD)
    print('=== %s (lambda_damp=%.3f, auto-leak=%.3f) ==='%(MD,lam,a._z_leak),flush=True)
    # (1) replay NRMSE
    a.reset(dt=DT); d=np.array([[float(v) for v in r] for r in list(csv.reader(open('/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv')))[1:]])
    Fp=[];Fr=[]
    for i in range(0,len(d),7):
        st=np.array([d[i,1],d[i,2],d[i,3],d[i,4]]); cl,cm=a.predict(st,d[i,8],d[i,7],U); Fp.append(cl*QD); Fr.append(d[i,5]); a.advance(st,d[i,8],d[i,7],U,DT)
    Fp=np.array(Fp);Fr=np.array(Fr); print('  replay NRMSE_F_y = %.3f'%(np.sqrt(np.mean((Fp-Fr)**2))/(Fr.max()-Fr.min())),flush=True)
    # (2) drive z to equilibrium from 0 (no input), report ||z||
    a.reset(dt=DT); x=np.zeros(4)
    for k in range(1500): a.advance(x,0.,0.,U,DT)
    zeq=np.linalg.norm(a._z); zsave=a._z.copy(); print('  free-run ||z|| equilibrium = %.1f'%zeq,flush=True)
    # (3) structural trim at z_eq, then gust vs no-gust coupling
    def rr(xe):
        a._z=zsave.copy(); cl,cm=a.predict(np.array([xe[0],0.,xe[1],0.]),0.,0.,U); return [-q*cl-structure.K_H*xe[0],q*cm*C-structure.K_ALPHA*xe[1]]
    xe=fsolve(rr,[0.,0.]); a._z=zsave.copy(); x0=np.array([xe[0],0.,xe[1],0.]); clt,cmt=a.predict(x0,0.,0.,U); CLt=float(clt); Fyt=q*CLt; Mzt=q*float(cmt)*C
    def openrun(wf):
        a._z=zsave.copy(); x=x0.copy(); tt=np.arange(0,1.6+DT,DT); CL=[];HD=[]
        for t in tt:
            W=wf(t); cl,cm=a.predict(x,0.,W,U); a.advance(x,0.,W,U,DT); Fy=q*cl-Fyt; Mz=q*cm*C-Mzt; _,hdd,_,_=structure.rhs(x,Fy,Mz); CL.append(cl);HD.append(hdd); x=structure.step_dp45(x,Fy,Mz,DT)
        return np.array(CL),np.array(HD)
    CLg,HDg=openrun(lambda t:gust(t)); CLn,HDn=openrun(lambda t:0.0)
    print('  trim C_L=%.3f  gust C_L excursion=%.3f'%(CLt,np.max(np.abs(CLg-CLt))),flush=True)
    print('  h_ddot peak: with gust=%.2f  no gust=%.2f  GUST EFFECT=%.2f (%s)'%(np.max(np.abs(HDg)),np.max(np.abs(HDn)),np.max(np.abs(HDg-HDn)),'COUPLED' if np.max(np.abs(HDg-HDn))>0.5 else 'decoupled'),flush=True)
