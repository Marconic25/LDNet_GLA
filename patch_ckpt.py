p='src/sensitivity_latent_damped_ckpt.py'
s=open(p).read()
anchor='    opt = optimization.OptimizationProblem(variables, loss_train, loss_valid)\n'
assert s.count(anchor)==1, ('anchor', s.count(anchor))
ins=anchor + '''    import json as _json
    _ck = RESULTS_DIR / ('latent_%d' % num_latent_states)
    _ck.mkdir(parents=True, exist_ok=True)
    _cfg = {'problem': problem, 'normalization': normalization, 'num_latent_states': num_latent_states, 'lambda_damp': LAMBDA_DAMP}
    with open(_ck / 'config.json', 'w') as _fp: _json.dump(_cfg, _fp, indent=2)
    def _save_ckpt(it):
        NNdyn.save_weights(str(_ck / 'NNdyn_weights.weights.h5'))
        NNrec.save_weights(str(_ck / 'NNrec_weights.weights.h5'))
        print('  [ckpt] saved weights at iter %d' % it, flush=True)
    opt.checkpoint_callback = _save_ckpt
    opt.checkpoint_every = 200
'''
s=s.replace(anchor, ins)
open(p,'w').write(s)
import ast; ast.parse(s)
print('CKPT PATCH OK')
