#!/usr/bin/env python3
"""
generate_ldnet_arch.py  v4
Layout matched to reference paper figure:
  - ∫dt → s(t) arrow goes DOWN-then-LEFT (below NN_dyn block)
  - s(t) is shared: sits between blocks, receives from ∫dt on the right
  - Orange rectangles around every s(t) neuron group (standalone + inside blocks)
"""
import os

# ─── Visual constants ──────────────────────────────────────────────────────────
D    = 16    # neuron diameter [px]
VGAP = 4     # gap between neurons [px]
NW   = 22    # normalisation strip width [px]
HGAP = 52    # horiz centre-to-centre between neuron columns [px]

# ─── Colours ───────────────────────────────────────────────────────────────────
C = {
    'u':    ('#67AB9F', '#3d7a6e'),
    'uinf': ('#a8d5c8', '#3d7a6e'),
    'lat':  ('#f0a30a', '#BD7000'),
    'x':    ('#6c8ebf', '#23445D'),
    'hid':  ('#ffffff', '#aaaaaa'),
    'oy':   ('#cc99ff', '#7B3FB0'),
}
TEAL   = '#3d7a6e'
ORANGE = '#BD7000'
BLUE   = '#23445D'
PURPLE = '#7B3FB0'

_id = [50]
lines = []

def uid(): _id[0] += 1; return _id[0]
def col_h(n): return n * D + (n - 1) * VGAP

# ─── Primitives ────────────────────────────────────────────────────────────────

def neuron(cx, cy, col='hid'):
    fc, sc = C[col]
    lines.append(
        f'<mxCell id="{uid()}" value="" '
        f'style="ellipse;fillColor={fc};strokeColor={sc};strokeWidth=1.5;html=1;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{cx-D/2:.1f}" y="{cy-D/2:.1f}" '
        f'width="{D}" height="{D}" as="geometry"/>'
        f'</mxCell>')

def neuron_col(cx, cen_y, n, col='hid'):
    tot = col_h(n); top = cen_y - tot / 2
    pts = []
    for i in range(n):
        cy = top + i * (D + VGAP) + D / 2
        neuron(cx, cy, col)
        pts.append((cx, cy))
    return pts

def weight_lines(L, R, step=1):
    for i, (lx, ly) in enumerate(L):
        for j, (rx, ry) in enumerate(R):
            if (i + j) % max(1, step) == 0:
                lines.append(
                    f'<mxCell id="{uid()}" value="" '
                    f'style="endArrow=none;strokeColor=#cccccc;strokeWidth=0.7;html=1;" '
                    f'edge="1" parent="1">'
                    f'<mxGeometry relative="1" as="geometry">'
                    f'<mxPoint x="{lx:.1f}" y="{ly:.1f}" as="sourcePoint"/>'
                    f'<mxPoint x="{rx:.1f}" y="{ry:.1f}" as="targetPoint"/>'
                    f'</mxGeometry></mxCell>')

def norm_strip(x, y, h):
    lines.append(
        f'<mxCell id="{uid()}" value="" '
        f'style="fillColor=#444444;strokeColor=none;html=1;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{NW}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>')
    tx = x + NW / 2; ty = y + h / 2
    lines.append(
        f'<mxCell id="{uid()}" value="normalization" '
        f'style="text;html=1;strokeColor=none;fillColor=none;align=center;'
        f'fontSize=8;fontColor=#ffffff;rotation=-90;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{tx - h/2:.1f}" y="{ty - 7:.1f}" '
        f'width="{h:.1f}" height="14" as="geometry"/>'
        f'</mxCell>')

def neuron_box(cx, cen_y, n, color=ORANGE):
    """Orange rectangle outline around a column of n neurons (no fill)."""
    PAD = 5
    bw = D + PAD * 2; bh = col_h(n) + PAD * 2
    bx = cx - D / 2 - PAD; by = cen_y - col_h(n) / 2 - PAD
    lines.append(
        f'<mxCell id="{uid()}" value="" '
        f'style="rounded=0;fillColor=none;strokeColor={color};strokeWidth=2;html=1;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{bx:.1f}" y="{by:.1f}" '
        f'width="{bw:.1f}" height="{bh:.1f}" as="geometry"/>'
        f'</mxCell>')

def lbl(val, x, y, w, h, st='fontSize=12;align=center;'):
    lines.append(
        f'<mxCell id="{uid()}" value="{val}" '
        f'style="text;html=1;strokeColor=none;fillColor=none;{st}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>')

def rect(val, x, y, w, h, st):
    lines.append(
        f'<mxCell id="{uid()}" value="{val}" style="{st}" vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>')

def arrow(pts, color, w=2.5, dashed=False):
    dash = 'dashed=1;dashPattern=8 4;' if dashed else ''
    src = pts[0]; tgt = pts[-1]; mid = pts[1:-1]
    wps = ''.join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in mid)
    lines.append(
        f'<mxCell id="{uid()}" value="" '
        f'style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor={color};'
        f'strokeWidth={w};endArrow=block;endFill=1;{dash}" '
        f'edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{src[0]:.1f}" y="{src[1]:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{tgt[0]:.1f}" y="{tgt[1]:.1f}" as="targetPoint"/>'
        f'<Array as="points">{wps}</Array>'
        f'</mxGeometry></mxCell>')

# ─── Architecture ──────────────────────────────────────────────────────────────
NU    = 6    # η(t): W_gust, δ, h, ḣ, α, α̇
NS    = 4    # standalone s(t) neurons (d_s = 10 in reality)
NH_D  = 7    # NN_dyn hidden
NO_D  = 4    # NN_dyn output (ds/dt)
NS_R  = 4    # s(t) fed into NN_rec
NX_R  = 2    # x spatial query
NH_R  = 6    # NN_rec hidden
NO_R  = 2    # NN_rec output: F_y, M_z

# ─── X layout ─────────────────────────────────────────────────────────────────
#
#  labels   s(t)  |left-norm|  inp   h1    h2   dyn-out  |right-norm|  ds/dt  ∫dt
#   10       90      140        168   220   272   324        346         374   450
#
XS      = 90       # standalone s(t) column centre-x
XNRM_L  = 140      # left norm strip left edge (both blocks)
XC0     = 168      # input column centre
XC1     = XC0 + HGAP   # 220
XC2     = XC1 + HGAP   # 272
XC3     = XC2 + HGAP   # 324  (dyn out / rec h3)
XC4     = XC3 + HGAP   # 376  (rec output)

XNRM_RD = int(XC3 + D / 2 + 8)    # right norm NN_dyn   (348)
XNRM_RR = int(XC4 + D / 2 + 8)    # right norm NN_rec   (400)

XARR_D  = XNRM_RD + NW + 10       # start of ds/dt label
XINTEG  = XARR_D + 70             # ∫dt oval left edge
INTEG_W = 60; INTEG_H = 60

XOUT_R  = XNRM_RR + NW + 14       # ỹ label x

# ─── Y layout ─────────────────────────────────────────────────────────────────
YDYN = 190     # NN_dyn content centre-y
YREC = 545     # NN_rec content centre-y

GAP14 = 18     # gap between u and s sub-groups inside block

# NN_dyn: u-group + gap + s-group + gap + U_inf
UINF_SEP = 12
inp_d_h  = col_h(NU) + GAP14 + col_h(NS) + UINF_SEP + D
DYN_H    = max(inp_d_h, col_h(NH_D), col_h(NO_D)) + 50
DYN_T    = YDYN - DYN_H // 2
DYN_BOT  = DYN_T + DYN_H

# sub-group y-centres inside NN_dyn
YU    = DYN_T + 30 + col_h(NU) / 2
YSd   = YU + col_h(NU) / 2 + GAP14 + col_h(NS) / 2
YUinf = YSd + col_h(NS) / 2 + UINF_SEP + D / 2

# s(t) standalone: positioned between NN_dyn bottom and NN_rec top,
# closer to NN_dyn so the ∫dt feedback arrow is short
YS    = DYN_BOT + 40 + col_h(NS) / 2    # standalone s(t) centre-y
S_TOP = YS - col_h(NS) / 2
S_BOT = YS + col_h(NS) / 2

# NN_rec
inp_r_h = col_h(NS_R) + GAP14 + col_h(NX_R)
REC_H   = max(inp_r_h, col_h(NH_R), col_h(NO_R)) + 50
# pin NN_rec top = S_BOT + gap
REC_T   = S_BOT + 50
YREC    = REC_T + REC_H // 2

YSr = REC_T + 30 + col_h(NS_R) / 2
YXr = YSr + col_h(NS_R) / 2 + GAP14 + col_h(NX_R) / 2

# ─── Build ─────────────────────────────────────────────────────────────────────
lines.append('<mxCell id="0"/>')
lines.append('<mxCell id="1" parent="0"/>')

CANVAS_W = max(XINTEG + INTEG_W + 80, XOUT_R + 220)
CANVAS_H = REC_T + REC_H + 60

# Title
lbl('LDNet &#8212; GLA Architecture', 60, 12, CANVAS_W - 120, 28,
    st='fontSize=16;fontStyle=1;align=center;fontColor=#333333;')

# s(0)=0
lbl('s(0) = &lt;b&gt;0&lt;/b&gt;', XS - 22, DYN_T - 20, 72, 16,
    st='fontSize=10;fontColor=#aaaaaa;align=center;fontStyle=2;')

# ── NN_DYN block ───────────────────────────────────────────────────────────────
BLK_W_D = XNRM_RD + NW - XNRM_L + 16
rect('', XNRM_L - 8, DYN_T, BLK_W_D, DYN_H,
     'rounded=1;arcSize=4;fillColor=#EBF5FB;strokeColor=#6c8ebf;strokeWidth=2;html=1;')
lbl('&lt;b&gt;NN&lt;sub&gt;dyn&lt;/sub&gt;&lt;/b&gt;&#160;&#160;'
    'w&lt;sub&gt;dyn&lt;/sub&gt;&#160;|&#160;tanh&#160;|&#160;7&#8211;7&#8211;d&lt;sub&gt;s&lt;/sub&gt;',
    XNRM_L, DYN_T + 6, BLK_W_D - 12, 16,
    st='fontSize=9;fontColor=#6c8ebf;align=left;')
norm_strip(XNRM_L, DYN_T + 26, DYN_H - 32)

u_col    = neuron_col(XC0, YU,    NU,   'u')
s_col_d  = neuron_col(XC0, YSd,   NS,   'lat')
neuron(XC0, YUinf, 'uinf')
uinf_pt  = [(XC0, YUinf)]
inp_d    = u_col + s_col_d + uinf_pt

h1_d = neuron_col(XC1, YDYN, NH_D, 'hid')
h2_d = neuron_col(XC2, YDYN, NH_D, 'hid')
o_d  = neuron_col(XC3, YDYN, NO_D, 'lat')

weight_lines(inp_d, h1_d)
weight_lines(h1_d,  h2_d)
weight_lines(h2_d,  o_d)
norm_strip(XNRM_RD, DYN_T + 26, DYN_H - 32)

# Orange box around s(t) input group inside NN_dyn
neuron_box(XC0, YSd, NS, ORANGE)

# ds/dt label + dashed line + ∫dt oval
lbl('&lt;b&gt;ds/dt&lt;/b&gt;', XARR_D - 2, YDYN - 11, 66, 22,
    st='fontSize=11;fontStyle=1;align=left;fontColor=#333333;')
rect('&#8747;dt', XINTEG, YDYN - INTEG_H // 2, INTEG_W, INTEG_H,
     'ellipse;html=1;strokeColor=#555555;fillColor=#f5f5f5;'
     'fontSize=18;fontStyle=1;align=center;verticalAlign=middle;')
arrow([(XNRM_RD + NW + 2, YDYN), (XINTEG, YDYN)], '#888888', w=1.5, dashed=True)

# ── standalone s(t) column ────────────────────────────────────────────────────
s_main = neuron_col(XS, YS, NS, 'lat')
# orange border box (the defining visual element of shared s(t))
neuron_box(XS, YS, NS, ORANGE)
lbl('&lt;b&gt;s&lt;/b&gt;(t)', XS - 26, S_TOP - 22, 72, 18,
    st='fontSize=13;fontStyle=1;fontColor=#f0a30a;align=center;')
lbl('&#8712;&#160;&lt;b&gt;&#8477;&lt;/b&gt;&lt;sup&gt;d&lt;sub&gt;s&lt;/sub&gt;&lt;/sup&gt;',
    XS - 20, S_BOT + 5, 58, 16,
    st='fontSize=10;fontColor=#BD7000;align=center;')

# ── NN_REC block ───────────────────────────────────────────────────────────────
BLK_W_R = XNRM_RR + NW - XNRM_L + 16
rect('', XNRM_L - 8, REC_T, BLK_W_R, REC_H,
     'rounded=1;arcSize=4;fillColor=#EBF5FB;strokeColor=#6c8ebf;strokeWidth=2;html=1;')
lbl('&lt;b&gt;NN&lt;sub&gt;rec&lt;/sub&gt;&lt;/b&gt;&#160;&#160;'
    'w&lt;sub&gt;rec&lt;/sub&gt;&#160;|&#160;tanh&#160;|&#160;24&#8211;24&#8211;24&#8211;24',
    XNRM_L, REC_T + 6, BLK_W_R - 12, 16,
    st='fontSize=9;fontColor=#6c8ebf;align=left;')
norm_strip(XNRM_L, REC_T + 26, REC_H - 32)

s_col_r = neuron_col(XC0, YSr, NS_R, 'lat')
x_col_r = neuron_col(XC0, YXr, NX_R, 'x')
inp_r   = s_col_r + x_col_r

h1_r = neuron_col(XC1, YREC, NH_R, 'hid')
h2_r = neuron_col(XC2, YREC, NH_R, 'hid')
h3_r = neuron_col(XC3, YREC, NH_R, 'hid')
o_r  = neuron_col(XC4, YREC, NO_R, 'oy')

weight_lines(inp_r, h1_r)
weight_lines(h1_r,  h2_r)
weight_lines(h2_r,  h3_r)
weight_lines(h3_r,  o_r)
norm_strip(XNRM_RR, REC_T + 26, REC_H - 32)

# Orange box around s(t) input group inside NN_rec
neuron_box(XC0, YSr, NS_R, ORANGE)

# ─── Output labels ─────────────────────────────────────────────────────────────
lbl('&lt;b&gt;&#7929;(&lt;b&gt;x&lt;/b&gt;,&#160;t)&lt;/b&gt;',
    XOUT_R, YREC - 32, 110, 26,
    st='fontSize=14;fontStyle=1;fontColor=#7B3FB0;align=left;')
lbl('F&lt;sub&gt;y&lt;/sub&gt;(t),&#160;M&lt;sub&gt;z&lt;/sub&gt;(t)',
    XOUT_R, YREC + 2, 130, 22,
    st='fontSize=11;fontColor=#7B3FB0;align=left;')
lbl('y&lt;sub&gt;0&lt;/sub&gt;&#160;&#8853;&#160;NN&lt;sub&gt;rec&lt;/sub&gt;',
    XOUT_R, YREC + 26, 130, 18,
    st='fontSize=9;fontColor=#aaaaaa;align=left;')

# ─── η(t) input labels ─────────────────────────────────────────────────────────
lbl('&lt;b&gt;&#951;&lt;/b&gt;(t)', 6, YU - col_h(NU)/2 - 22, 76, 18,
    st='fontSize=13;fontStyle=1;fontColor=#446359;align=center;')
names_u = ['W&lt;sub&gt;gust&lt;/sub&gt;', '&#948;', 'h', 'h&#775;', '&#945;', '&#945;&#775;']
for k, (ux, uy) in enumerate(u_col):
    lbl(names_u[k], 6, uy - 8, 70, 16, st='fontSize=9;fontColor=#446359;align=right;')
lbl('U&lt;sub&gt;&#8734;&lt;/sub&gt;', 6, YUinf - 8, 70, 16,
    st='fontSize=9;fontColor=#3d7a6e;align=right;fontStyle=2;')

# ─── x query box ───────────────────────────────────────────────────────────────
xbox_w = 80; xbox_h = 52
xbox_x = 6; xbox_y = YXr - xbox_h / 2
rect('&lt;b&gt;x&lt;/b&gt;&#160;&#8712;&#160;&#937;&#160;&#8834;&#160;'
     '&lt;b&gt;&#8477;&lt;/b&gt;&lt;sup&gt;d&lt;/sup&gt;',
     xbox_x, xbox_y, xbox_w, xbox_h,
     'rounded=1;html=1;strokeColor=#82b366;fillColor=#d5e8d4;'
     'fontSize=10;fontStyle=1;align=center;verticalAlign=middle;')

# ─── ARROWS ────────────────────────────────────────────────────────────────────

# 1. η(t) → left norm strip (one arrow per neuron)
for ux, uy in u_col:
    arrow([(78, uy), (XNRM_L + 1, uy)], TEAL, w=1.5)
# U_inf
arrow([(78, YUinf), (XNRM_L + 1, YUinf)], '#3d7a6e', w=1.5)

# 2. s(t) standalone → NN_dyn (right side of s(t) box → enter block at YSd level)
#    horizontal, goes from XS+box_right to XNRM_L
BOX_PAD = 5
arrow([(XS + D/2 + BOX_PAD + 2, YSd),
       (XNRM_L + 1, YSd)], ORANGE, w=2.5)

# 3. s(t) standalone → NN_rec (exits bottom of s(t) box, goes down to YSr)
#    Exits bottom-centre, goes down then right to enter NN_rec at YSr level.
arrow([(XS, S_BOT + BOX_PAD + 2),
       (XS, YSr),
       (XNRM_L + 1, YSr)], ORANGE, w=2.5)

# 4. ∫dt → s(t) feedback
#    Route: ∫dt bottom → go DOWN to just below NN_dyn block → LEFT to XS → UP to s(t) top
#    This goes BELOW NN_dyn (no crossing) and arrives at s(t) from above.
INTEG_CX   = XINTEG + INTEG_W / 2
INTEG_BOT  = YDYN + INTEG_H / 2
BELOW_DYN  = DYN_BOT + 16            # y just below NN_dyn block
arrow([
    (INTEG_CX, INTEG_BOT),            # ∫dt bottom centre
    (INTEG_CX, BELOW_DYN),            # go DOWN below NN_dyn
    (XS,       BELOW_DYN),            # go LEFT to s(t) column
    (XS,       S_TOP - BOX_PAD - 2),  # arrive at top of s(t) box
], ORANGE, w=2.5)

# 5. x query → NN_rec input
arrow([(xbox_x + xbox_w, YXr), (XNRM_L + 1, YXr)], BLUE, w=2.0)

# 6. NN_rec output → ỹ label
arrow([(XNRM_RR + NW + 2, YREC), (XOUT_R - 2, YREC)], PURPLE, w=2.0)

# ─── Assemble XML ───────────────────────────────────────────────────────────────
xml = f'''<mxfile host="app.diagrams.net" version="21.7.5">
  <diagram id="ldnet-arch" name="LDNet Architecture">
    <mxGraphModel dx="1200" dy="800" grid="0" gridSize="10"
      guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0"
      pageScale="1" pageWidth="{CANVAS_W}" pageHeight="{CANVAS_H}"
      math="0" shadow="0">
      <root>
        {''.join(lines)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs', 'ldnet_arch.drawio')
with open(out, 'w', encoding='utf-8') as f:
    f.write(xml)
print(f'Written → {out}')
print(f'  Canvas: {CANVAS_W} x {CANVAS_H} px')
print(f'  Elements: {_id[0] - 50}')
