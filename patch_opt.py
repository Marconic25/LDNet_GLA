import re
p='src/optimization.py'
s=open(p).read()
assert 'checkpoint_callback' not in s, 'already patched'
a='        self.iteration = 0\n        self.iterations_history = list()'
b='        self.iteration = 0\n        self.checkpoint_callback = None\n        self.checkpoint_every = 200\n        self.iterations_history = list()'
assert s.count(a)==1, ('init anchor count', s.count(a))
s=s.replace(a,b)
c="                  (self.iteration, self.loss_train_history[-1], self.loss_valid_history[-1]))\n        self.iteration += 1"
d="                  (self.iteration, self.loss_train_history[-1], self.loss_valid_history[-1]))\n        if self.checkpoint_callback is not None and self.iteration > 0 and self.iteration % self.checkpoint_every == 0:\n            self.checkpoint_callback(self.iteration)\n        self.iteration += 1"
assert s.count(c)==1, ('cb anchor count', s.count(c))
s=s.replace(c,d)
open(p,'w').write(s)
import ast; ast.parse(s)
print('PATCH OK')
