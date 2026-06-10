import numpy as np, structure, json, os
from ldnet_aero import LDNetAero
U=80.;RHO=1.225;S=0.05;C=1.0;DT=0.002;q=0.5*RHO*U**2*S
for MD in ['models_damped_l003_full/latent_10','models_damped_l01_full/latent_10','models_damped/latent_10']:
    a=LDNetAero(MD); a.reset(dt=DT); x=np.zeros(4)
    print('=== %s : free-run joint relaxation timeline ==='%MD,flush=True)
    for k in range(1,3001):
        cl,cm=a.predict(x,0.,0.,U); a.advance(x,0.,0.,U,DT); x=structure.step_rk4(x,q*cl,q*cm*C,DT)
        if k in (100,250,500,750,1000,1500,2000,3000):
            print('  step %4d (t=%.2fs): ||z||=%7.1f  C_L=%7.3f  h=%8.4f'%(k,k*DT,np.linalg.norm(a._z),float(cl),x[0]),flush=True)
