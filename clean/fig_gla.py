import numpy as np, csv, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import structure
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; S=0.05; C=1.0; DT=0.002; q=0.5*RHO*U**2*S
DS='/work/u10677113/NACA2312/dataset_v5'; MD='models_rollout/latent_10'; SIM='sim_A_025_test'
DMAX=14.0; DDOT=300.0
d=np.array([[float(v) for v in r] for r in list(csv.reader(open(f'{DS}/{SIM}/structural_trajectory.csv')))[1:]])
d=d[range(0,len(d),7)]; W=d[:,7]; t=d[:,0]; N=len(d); x0=np.array([d[0,1],d[0,2],d[0,3],d[0,4]])
a=LDNetAero(MD); a.reset(dt=DT); clt,_=a.predict(x0,0.,0.,U); CLTRIM=float(clt)
def run(gain):
    a.reset(dt=DT); x=x0.copy(); dprev=0.; CL=[];HH=[];DD=[]
    for i in range(N):
        clm,_=a.predict(x,dprev,W[i],U)
        dcmd=-gain*(float(clm)-CLTRIM); dcmd=np.clip(dcmd,dprev-DDOT*DT,dprev+DDOT*DT)
        dcmd=float(np.clip(dcmd,-DMAX,DMAX))
        cl,cm=a.predict(x,dcmd,W[i],U); a.advance(x,dcmd,W[i],U,DT); x=structure.step_rk4(x,q*cl,q*cm*C,DT)
        CL.append(float(cl));HH.append(x[0]);DD.append(dcmd); dprev=dcmd
    return np.array(CL),np.array(HH),np.array(DD)
CLo,Ho,_=run(0.0); CLc,Hc,Dc=run(20.0)
m=t<=1.8
fig,ax=plt.subplots(1,3,figsize=(15,4))
fig.suptitle('GLA with rollout-trained LDNet (closed loop, real gust sim_A_025) — proportional gain=20',fontsize=12)
ax[0].plot(t[m],CLo[m],'b',lw=1.8,label='open loop'); ax[0].plot(t[m],CLc[m],'r',lw=1.8,label='controlled')
ax[0].axhline(CLTRIM,color='gray',ls='--',lw=1); ax[0].set_xlabel('t [s]');ax[0].set_ylabel('$C_L$');ax[0].set_title('Lift coefficient (-59%% excursion)');ax[0].legend();ax[0].grid(alpha=.3)
ax[1].plot(t[m],Ho[m]*1000,'b',lw=1.8,label='open');ax[1].plot(t[m],Hc[m]*1000,'r',lw=1.8,label='controlled')
ax[1].set_xlabel('t [s]');ax[1].set_ylabel('h [mm]');ax[1].set_title('Heave');ax[1].legend();ax[1].grid(alpha=.3)
ax[2].plot(t[m],Dc[m],'r',lw=1.8);ax[2].set_xlabel('t [s]');ax[2].set_ylabel('flap $\\delta$ [deg]');ax[2].set_title('Control flap');ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig('results/fig_gla_rollout.png',dpi=130)
print('saved results/fig_gla_rollout.png',flush=True)
print('open exc=%.3f controlled exc=%.3f'%(np.max(np.abs(CLo[m]-CLTRIM)),np.max(np.abs(CLc[m]-CLTRIM))),flush=True)
