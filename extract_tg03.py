import numpy as np

for W0 in [10, 20, 30]:
    d = np.load(f'light/results_cs25_combo/traces_W{W0}.npz')
    tag = 'Tg0.30'
    keys = [k for k in d.files if k.startswith(tag)]
    if not keys:
        print(W0, 'NO Tg0.30 DATA')
        continue
    jb = int(d[f'{tag}_jb'])
    clred = float(d[f'{tag}_clred'])
    fmax = float(d[f'{tag}_fmax'][jb])
    print(W0, 'Tg=0.30', 'CLred%=', round(clred, 1), 'fmax(deg)=', round(fmax, 2))
    print('  all keys for this tag:', keys)
