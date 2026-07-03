# 04 — Equal information, and the chatter/smoothing reversal

[← Back to index](README.md) · [← 03 Oracle & observability](03-gust-oracle-and-observability.md)

Two experiments here, both aimed at removing the *unfair advantages* the optimal
had been enjoying: (A) the extra information (`W`), and (B) the comparison against
an un-smoothed proportional.

## A. Equal-information test — both controllers know `W`

If the optimal's edge was *information* (`W`), then giving the **proportional** the
same information — via a simple gust **feedforward** — should close the gap. If the
optimal still wins at equal information, then the model/optimization itself is
adding value.

Script: `clean/propw.py`. Three arms, all fair on the sensor, compared across the
**full in-envelope grid** `W in {10, 20, 30} × Tg in {0.30, 0.40, 0.50, 0.70, 1.00, 1.20}`:

| Arm | Law | Information used |
|-----|-----|------------------|
| `prop-CL` | `delta = g_CL*(C_L_meas - trim)` | `C_L` only (reactive) |
| `prop-W` | `delta = g_CL*(C_L_meas - trim) + g_W*W(t)` | `C_L` **+ gust feedforward** |
| `opt-W` | single-step optimal with the `W` oracle | model + optimization + `W` |

Both `prop-W` and `opt-W` are handed the true `W`, so the comparison isolates
*controller sophistication* at equal information. The metric is max `C_L` reduction
with pitch kept `<=` open-loop.

**Result:**

> `prop-W ≈ opt-W` across most of the envelope — margins `~ ±5` points. The optimal
> wins **clearly** only in a **few high-`k` / strong** cells, and even there the
> result is **noisy and cell-dependent**, *not* a clean monotone `k`-trend.

`margin = opt-W − prop-W ≈ 0` means the model's advantage was **the information**
(`W`), which a trivial linear **feedforward** term realizes — not the model or the
optimization.

> **Walked-back reading.** An initial pass *looked* like "margin grows with `k`,"
> which would have vindicated the reduced-frequency prediction from
> [03](03-gust-oracle-and-observability.md). On inspection this was an artifact of
> how the winning arm was **picked** under the pitch constraint (a pitch-fallback
> pick), not a genuine trend. See [06 — pitfalls](06-bugs-and-pitfalls.md). The
> honest statement is: *the optimal wins only at a few noisy strong/high-`k` cells,
> with no clean `k`-law.*

## B. The chatter/smoothing reversal

Every oracle-win so far compared the optimal against a **raw, unsmoothed**
proportional. Was the optimal's "smoothness + extra `C_L` reduction" a real
advantage, or an artifact of that unfair baseline?

### The chatter

The raw high-gain proportional **chatters**: a high-gain output-feedback limit
cycle in which the flap slews at the rate limit back and forth. Quantitatively, its
**total variation of `delta`** over the gust window is huge:

| Controller | TV(delta) [deg] |
|------------|-----------------|
| Proportional, raw high gain | ~200–500 (limit-cycle chatter) |
| Optimal | ~15–65 (smooth) |

So on a raw comparison the optimal looks vastly smoother — but that is comparing a
**deliberately** smoothed model controller to an **un**-filtered proportional.

### The fix: smooth the proportional too

Script: `clean/smoothprop.py`. Add a simple **2nd-order (cascaded) low-pass** to the
proportional command (`DLPF ≈ 0.7`), the *same* smoothing chain the MPC arm uses,
and re-sweep. Result:

| Arm | `C_L` reduction | TV(delta) |
|-----|-----------------|-----------|
| prop raw | baseline | ~200–500 (chatter) |
| **prop + LPF (`DLPF≈0.7`)** | **higher than raw** | low |
| opt | comparable | ~15–65 |

> The low-pass **removes the chatter AND *increases* the proportional's `C_L`
> reduction.** The chatter had been **wasting control authority** — slewing the flap
> back and forth at the rate limit instead of putting it where it alleviates lift.
> Once smoothed, the proportional **matches or beats** the optimal at **equal (low)
> chatter**.

### The takeaway

The optimal's apparent "smoothness + extra `C_L` reduction" advantage was **largely
an artifact of comparing against an un-smoothed proportional.** At equal smoothing,
the two are on par (and the proportional often wins on `C_L` reduction). The optimal
does not have a genuine smoothness edge — it just had a smoothing stage its
competitor was denied.

Note the harness makes this an *apples-to-apples* filter comparison by design: in
`clean/mpc_gust.py::simulate`, both `mpc` and `prop` arms pass through the **same**
2nd-order flap-smoothing chain (`de_f`, `de_f2` with `DLPF`) and the **same**
saturation (14 deg) / rate limit (300 deg/s), so any remaining difference is the
control law, not the post-filter.

## Combined conclusion of this section

At **equal information** (both fed `W`) **and equal smoothing** (both low-passed):

- prop-with-feedforward ≈ optimal across most of the envelope;
- the smoothed proportional matches/beats the optimal on `C_L` reduction at equal
  chatter.

The two "wins" the optimal appeared to have — extra reduction on sharp gusts, and
smoothness — both dissolve once the comparison is made fair. What remains is a
handful of noisy strong/high-`k` cells, and (as the next section shows) at the very
edge of the envelope the optimal actually **loses**.

---

Next: [05 — Envelope edge and open questions →](05-envelope-edge-and-open-questions.md)
