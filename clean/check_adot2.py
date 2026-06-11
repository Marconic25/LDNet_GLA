import numpy as np, csv
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; C=1.0; DT=0.002; S=0.05; q=0.5*RHO*U**2*S
MD='models_rollout/latent_10'
X0=np.array([-6.49179e-3,0.,-8.76338e-4,0.])
a=LDNetAero(MD)
def openrun(Wt):                 # returns alpha_dot STATE (x[3]) trajectory
    a.reset(dt=DT); x=X0.copy(); AD=[]
    for W in Wt:
        cl,cm=a.predict(x,0.,W,U); a.advance(x,0.,W,U,DT); x=structure.step_rk4(x,q*cl,q*cm*C,DT)
        AD.append(x[3])
    return np.array(AD)
def ring(ad,t,t0,t1):
    m=(t>=t0)&(t<=t1); s=ad[m]
    return np.sqrt(np.mean(s**2)), int(np.sum(np.abs(np.diff(np.sign(s-np.mean(s)))))/2)
TG=1.12;W0=11.46;N=1500;t=np.arange(N)*DT
Ws=np.array([(W0/2)*(1-np.cos(2*np.pi*tt/TG)) if 0<=tt<=TG else 0. for tt in t])
ad_s=openrun(Ws)
d=np.array([[float(v) for v in r] for r in list(csv.reader(open('/work/u10677113/NACA2312/dataset_v5/sim_A_025_test/structural_trajectory.csv')))[1:]])
d=d[range(0,len(d),7)];Wd=d[:,7];td=d[:,0]
ad_dm=openrun(Wd); ad_cfd=d[:,4]
DEG=180/np.pi
print('alpha_dot STATE (deg/s) ringing, open-loop:',flush=True)
print('  synth gust MODEL: during RMS=%.4g xc=%d | post[1.4,2.4] RMS=%.4g'%(ring(ad_s*DEG,t,0,1.12)[0],ring(ad_s,t,0,1.12)[1],ring(ad_s*DEG,t,1.4,2.4)[0]),flush=True)
print('  data  gust MODEL: during RMS=%.4g xc=%d | post RMS=%.4g'%(ring(ad_dm*DEG,td,0,1.12)[0],ring(ad_dm,td,0,1.12)[1],ring(ad_dm*DEG,td,1.4,2.4)[0]),flush=True)
print('  data  gust CFD  : during RMS=%.4g xc=%d | post RMS=%.4g'%(ring(ad_cfd*DEG,td,0,1.12)[0],ring(ad_cfd,td,0,1.12)[1],ring(ad_cfd*DEG,td,1.4,2.4)[0]),flush=True)
print('  peaks deg/s: synthMODEL=%.3f dataMODEL=%.3f dataCFD=%.3f'%(np.max(np.abs(ad_s))*DEG,np.max(np.abs(ad_dm))*DEG,np.max(np.abs(ad_cfd))*DEG),flush=True)
