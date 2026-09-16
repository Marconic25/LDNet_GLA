import numpy as np

U = 80.0
Tg = 0.30

for W0 in [10, 20, 30]:
    d = np.load(f'light/results_cs25_combo/traces_W{W0}.npz')
    tag = 'Tg0.30'
    jb = int(d[f'{tag}_jb'])
    clred = float(d[f'{tag}_clred'])
    fmax = float(d[f'{tag}_fmax'][jb])
    Rstar = float(d[f'{tag}_Rstar'])
    pitch_arr = np.atleast_1d(d[f'{tag}_pitch'])
    pitch = float(pitch_arr[jb]) if pitch_arr.size > 1 else float(pitch_arr[0])
    H = U * Tg / 2.0 / 0.3048
    k = np.pi * 1.0 / (U * Tg)  # c_ref = 1 m per chapter3.tex Table (structural parameters)
    print(f'W0={W0} Tg={Tg} H={H:.0f} k={k:.3f} CLred%={clred:.1f} Rstar={Rstar:.0e} fmax={fmax:.2f} pitch={pitch:.2f}')
