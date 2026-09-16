import io
p = "final/cluster/run_sim.pbs"
s = io.open(p, encoding="utf-8").read()
old = """#   Windows must stay a MULTIPLE OF 29: other values misalign OpenFOAM's write
#   cadence with the driver's restart times and the solver dies looking for a
#   time directory that was never written (window=50 failed exactly this way).
"""
new = """#   Measured confirmation at window 63 (job 31470, sim_Cc_041_train): the MPC
#   moves the flap 0.525 deg per window -- the exact step dagger records as
#   "mediana E massimo" -- which over 1.991 ms is 263.7 deg/s, against dagger's
#   validated 258.6 deg/s and under the 300 deg/s limit. The same 0.525 deg over
#   window 29's 0.916 ms would have been 572.9 deg/s.
#
#   NOTE: window=50 crashed with "cannot find processor*/<t>/p" and I guessed the
#   window had to be a multiple of 29. That guess is WRONG -- 63 is not a multiple
#   of 29 and runs fine. The window=50 failure has some other, still unidentified
#   cause; do not treat multiples of 29 as a constraint.
"""
assert old in s
io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("comment corrected")
