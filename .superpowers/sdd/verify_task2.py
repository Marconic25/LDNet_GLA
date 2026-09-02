import ast, sys

# Check parse
with open('/home/marco/LDNet_OF/light/run.py') as f:
    src = f.read()
ast.parse(src)
print('parse OK')

# Check signature
sys.path.insert(0, '/home/marco/LDNet_OF/light')
import inspect

# We can't do a full import (needs TF/LDNet), but we can parse and check
# Use ast to verify parameters
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'simulate':
        args = [a.arg for a in node.args.args]
        defaults_kw = [a.arg for a in node.args.args[len(node.args.args)-len(node.args.defaults):]]
        all_args = args
        assert 'NH' in all_args, f'NH param missing, got: {all_args}'
        assert 'R_du' in all_args, f'R_du param missing, got: {all_args}'
        print(f'simulate() signature OK: {all_args}')
        break

# Check Nsteps rename (no bare N = int(round) left)
assert 'N  = int(round' not in src, 'Old bare N variable still present!'
assert 'N = int(round' not in src, 'Old bare N variable still present!'
assert 'Nsteps = int(round' in src, 'Nsteps rename missing!'
print('N->Nsteps rename OK')

# Check MPCPreviewController import
assert 'from optimal import OptimalController, MPCPreviewController' in src
print('MPCPreviewController import OK')

# Check LAM
assert 'LAM    = float(aero._z_leak)' in src
print('LAM module-level OK')

# Check combo controller construction
assert "elif mode == 'combo':" in src
assert 'MPCPreviewController(' in src
print('combo controller construction OK')

# Check combo loop branch
assert 'w_seq = np.zeros(NH)' in src
assert 'w_seq[:hi - lo] = Wt[lo:hi]' in src
print('combo loop branch OK')

# Check __main__ MODE/NH/R_DU env vars
assert "MODE  = os.environ.get('MODE', 'optimal')" in src
assert "NH    = int(os.environ.get('NH',   '8'))" in src
assert "R_DU  = float(os.environ.get('R_DU', '0.0'))" in src
print('__main__ env vars OK')

# Check open/optimal paths unchanged (critical invariant)
assert "ctrl = OptimalController(\n            aero, U=U, dt=DT, R=R, n_grid=NGRID," in src
print('optimal path unchanged OK')

print('\nAll checks passed.')
