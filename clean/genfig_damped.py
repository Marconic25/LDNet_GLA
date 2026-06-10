import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldnet_aero import LDNetAero
from pathlib import Path
OUT=Path('results'); U=80.;DT=0.002
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
for MD,lab,col in [('models/latent_10','original (lambda=0): DRIFTS','darkorange'),('models_damped/latent_10','damped (lambda=0.01): BOUNDED','crimson')]:
    a=LDNetAero(MD); a.reset(dt=DT); x=np.zeros(4); zn=[]
    for k in range(4000): a.advance(x,0.,0.,U,DT); zn.append(np.linalg.norm(a._z))
    tt=np.arange(len(zn))*DT
    ax[0].plot(tt,zn,col,lw=2,label=lab)
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('||z|| (latent norm)'); ax[0].set_title('Free-running latent norm (no input)')
ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
ax[1].plot(tt,zn,'crimson',lw=2); ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('||z||')
ax[1].set_title('Damped model zoomed: converges to equilibrium ~175'); ax[1].grid(alpha=.3); ax[1].set_ylim(0,400)
fig.suptitle('Latent stabilization: damped ODE retraining fixes the unbounded drift',fontsize=13)
fig.tight_layout(); fig.savefig(OUT/'figF_damped_stability.png',dpi=140); print('figF done',flush=True)
