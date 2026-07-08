import numpy as np, structure, json, os
from ldnet_aero import LDNetAero
U=80.; RHO=1.225; S=0.05; C=1.0; DT=0.002; q=0.5*RHO*U**2*S
def gust(t,W0=10.): return (W0/2)*(1-np.cos(2*np.pi*t/1.0)) if 0<=t<=1.0 else 0.0
MD=os.environ.get('MD','models_damped/latent_10')
a=LDNetAero(MD)
print('=== flap authority probe: %s ==='%MD,flush=True)
# settle joint equilibrium (no gust, no flap)
a.reset(dt=DT); x=np.zeros(4)
for _ in range(6000):
    cl,cm=a.predict(x,0.,0.,U); a.advance(x,0.,0.,U,DT); x=structure.step_dp45(x,q*cl,q*cm*C,DT)
xs=x.copy(); zs=a._z.copy(); cls,cms=a.predict(xs,0.,0.,U)
print('joint eq: C_L=%.4f C_M=%.4f ||z||=%.1f'%(float(cls),float(cms),np.linalg.norm(zs)),flush=True)
# (A) STATIC flap authority: hold delta const from eq, let z+struct re-settle, measure dC_L,dC_M and h_ddot transient
print('--- (A) static flap: dC_L, dC_M vs delta (re-settled), peak h_ddot transient ---',flush=True)
for dd in [-3.,-1.,1.,3.]:
    a._z=zs.copy(); x=xs.copy(); hddmax=0.
    for k in range(1500):
        cl,cm=a.predict(x,dd,0.,U); d=structure.rhs(x,q*cl,q*cm*C); hddmax=max(hddmax,abs(d[1]))
        a.advance(x,dd,0.,U,DT); x=structure.step_dp45(x,q*cl,q*cm*C,DT)
    clf,cmf=a.predict(x,dd,0.,U)
    print('  delta=%+.0f deg -> dC_L=%+.4f dC_M=%+.4f  peak h_ddot=%.3f'%(dd,float(clf)-float(cls),float(cmf)-float(cms),hddmax),flush=True)
# (B) FEEDFORWARD ideal: best constant flap to cancel gust C_L excursion
print('--- (B) gust + constant feedforward flap: C_L excursion vs delta_ff ---',flush=True)
def run_ff(dff):
    a._z=zs.copy(); x=xs.copy(); CL=[];HD=[]
    for t in np.arange(0,1.6+DT,DT):
        W=gust(t); cl,cm=a.predict(x,dff,W,U); d=structure.rhs(x,q*cl,q*cm*C); CL.append(float(cl));HD.append(d[1])
        a.advance(x,dff,W,U,DT); x=structure.step_dp45(x,q*cl,q*cm*C,DT)
    CL=np.array(CL);HD=np.array(HD); return np.max(np.abs(CL-float(cls))),np.max(np.abs(HD))
ex0,hd0=run_ff(0.)
print('  no flap:        C_L excursion=%.4f  h_ddot peak=%.3f'%(ex0,hd0),flush=True)
for dff in [-2.,-1.,-0.5,0.5,1.,2.]:
    ex,hd=run_ff(dff); print('  delta_ff=%+.1f:  C_L excursion=%.4f (%+.1f%%)  h_ddot peak=%.3f'%(dff,ex,(ex0-ex)/ex0*100,hd),flush=True)
