import numpy as np, csv, json
from ldnet_aero import LDNetAero
U=80.;DT=0.002;QD=0.5*1.225*U**2*0.05
for MD in ['models/latent_10','models_damped/latent_10']:
    cfg=json.load(open(MD+'/config.json')); lam=cfg.get('lambda_damp',0.0)
    a=LDNetAero(MD); a.reset(dt=DT); a._z_leak=lam
    # (1) replay NRMSE on sim_A_025_test
    d=np.array([[float(v) for v in r] for r in list(csv.reader(open('/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv')))[1:]])
    Fp=[];Fr=[]
    for i in range(0,len(d),7):
        st=np.array([d[i,1],d[i,2],d[i,3],d[i,4]]); cl,cm=a.predict(st,d[i,8],d[i,7],U); Fp.append(cl*QD); Fr.append(d[i,5]); a.advance(st,d[i,8],d[i,7],U,DT)
    Fp=np.array(Fp);Fr=np.array(Fr); nr=np.sqrt(np.mean((Fp-Fr)**2))/(Fr.max()-Fr.min())
    # (2) free-running stability: no teacher forcing, zero structural input, no gust, 3000 steps
    a.reset(dt=DT); a._z_leak=lam; x=np.zeros(4); zn=[]
    for k in range(3000):
        cl,cm=a.predict(x,0.,0.,U); a.advance(x,0.,0.,U,DT); zn.append(np.linalg.norm(a._z))
    zn=np.array(zn)
    print('%-22s lambda=%.3f  replay_NRMSE_Fy=%.3f  free-run ||z||: @100=%.1f @1000=%.1f @3000=%.1f  %s'%(MD,lam,nr,zn[100],zn[1000],zn[-1],'BOUNDED' if zn[-1]<2*zn[1000]+50 else 'DRIFTING'),flush=True)
