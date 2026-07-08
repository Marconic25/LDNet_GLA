#!/usr/bin/env python3
"""
generate_ldnet_arch.py
Generates figs/ldnet_arch.drawio — LDNet GLA architecture diagram
styled after the reference paper figure: individual neuron circles,
gray weight lines, dark normalization strips, color-coded signal flows.

Usage:
    python3 generate_ldnet_arch.py
    # opens figs/ldnet_arch.drawio in VS Code draw.io extension
"""
import os, math

# ─── Visual constants ─────────────────────────────────────────────────────────
D    = 16    # neuron circle diameter [px]
VGAP = 4     # gap between neuron circles [px]
HGAP = 52    # center-to-center horizontal spacing between columns [px]
NW   = 26    # normalization strip width [px]

# ─── Color palette ─────────────────────────────────────────────────────────────
C = {
    'u':   ('#67AB9F', '#446359'),  # teal:   u(t) input neurons
    'lat': ('#f0a30a', '#BD7000'),  # orange: s(t) latent state
    'x':   ('#6c8ebf', '#23445D'),  # blue:   x spatial query
    'hid': ('#ffffff', '#bbbbbb'),  # white:  hidden neurons
    'oy':  ('#d9b3ff', '#7B3FB0'),  # purple: ỹ output neurons
}

_id = [50]
lines = []

def uid():
    _id[0] += 1; return _id[0]

# ─── Drawing primitives ────────────────────────────────────────────────────────

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
    """Draw n neurons vertically centered at cen_y. Returns list of (cx, cy)."""
    tot = n * D + (n - 1) * VGAP
    top = cen_y - tot / 2
    pts = []
    for i in range(n):
        cy = top + i * (D + VGAP) + D / 2
        neuron(cx, cy, col)
        pts.append((cx, cy))
    return pts

def weight_lines(L, R, step=1):
    """Gray weight connection lines from column L to R (subset for clarity)."""
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
    """Dark normalization strip with rotated 'normalization' label."""
    lines.append(
        f'<mxCell id="{uid()}" value="" '
        f'style="fillColor=#555555;strokeColor=none;html=1;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" '
        f'width="{NW}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>')
    tx = x + NW / 2; ty = y + h / 2
    lines.append(
        f'<mxCell id="{uid()}" value="normalization" '
        f'style="text;html=1;strokeColor=none;fillColor=none;align=center;'
        f'fontSize=9;fontColor=#ffffff;rotation=-90;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{tx - h/2:.1f}" y="{ty - 8:.1f}" '
        f'width="{h:.1f}" height="16" as="geometry"/>'
        f'</mxCell>')

def lbl(val, x, y, w, h, st='fontSize=13;align=center;'):
    lines.append(
        f'<mxCell id="{uid()}" value="{val}" '
        f'style="text;html=1;strokeColor=none;fillColor=none;{st}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>')

def rect(val, x, y, w, h, st):
    lines.append(
        f'<mxCell id="{uid()}" value="{val}" style="{st}" vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>')

def arrow(pts, color, w=2.5, dashed=False):
    """Orthogonal arrow through a list of (x,y) waypoints."""
    dash = 'dashed=1;' if dashed else ''
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

# ─── Architecture parameters ───────────────────────────────────────────────────
NU    = 6   # u(t) input neurons: W_gust, δ, h, ḣ, α, α̇
NS    = 4   # s(t) neurons shown (representative; true d_s=10)
NH_D  = 7   # NN_dyn hidden layer size
NO_D  = 5   # NN_dyn output neurons (ds/dt)
NS_R  = 4   # s(t) neurons fed into NN_rec
NX_R  = 2   # x spatial query neurons
NH_R  = 6   # NN_rec hidden layer size
NO_R  = 2   # NN_rec output: F_y, M_z

def col_h(n):
    return n * D + (n - 1) * VGAP

# ─── Canvas layout ─────────────────────────────────────────────────────────────
#
#  x-axis:
#   XS     = 68   s(t) column center
#   XNORM  = 95   left norm strip left edge
#   XC0    = 132  input column center  (inside NN blocks)
#   XC1    = XC0 + HGAP  hidden-1
#   XC2    = XC1 + HGAP  hidden-2
#   XC3D   = XC2 + HGAP  output (NN_dyn)
#   XC3R   = XC2 + HGAP  hidden-3 (NN_rec)
#   XC4R   = XC3R+ HGAP  output (NN_rec)
#
#  y-axis:
#   YDYN   = 207  NN_dyn content center y
#   YS     = 360  s(t) column center y
#   YREC   = 530  NN_rec content center y

XS   = 68
XNRM = 95    # left norm strip left edge (shared dyn/rec)
XC0  = 134   # input column center
XC1  = XC0 + HGAP
XC2  = XC1 + HGAP
XC3D = XC2 + HGAP   # NN_dyn output
XC3R = XC2 + HGAP   # NN_rec hidden-3
XC4R = XC3R + HGAP  # NN_rec output

XNRM_RD = int(XC3D + D/2 + 6)    # right norm strip x (dyn)
XNRM_RR = int(XC4R + D/2 + 6)    # right norm strip x (rec)

XDSDT  = XNRM_RD + NW + 8        # ds/dt label x
XINTEG = XDSDT + 72               # ∫dt block x
XOUT_R = XNRM_RR + NW + 8        # F_y/M_z label x

YDYN = 207
YS   = 362
YREC = 530

# Block heights (add vertical padding)
DYN_H = max(col_h(NU) + 14 + col_h(NS), col_h(NH_D), col_h(NO_D)) + 36
REC_H = max(col_h(NS_R) + 14 + col_h(NX_R), col_h(NH_R), col_h(NO_R)) + 36
DYN_T = YDYN - DYN_H // 2
REC_T = YREC - REC_H // 2

# y-centers of input sub-groups inside NN_dyn
GAP14 = 14   # gap between u-group and s-group
YU = YDYN - (col_h(NS) + GAP14) / 2 - col_h(NU) / 2 + col_h(NU) / 2
YU = YDYN - (col_h(NU) + GAP14 + col_h(NS)) / 2 + col_h(NU) / 2
YSd = YU + col_h(NU) / 2 + GAP14 + col_h(NS) / 2  # s input center (dyn)
# Similar for rec
YSr = YREC - (col_h(NS_R) + GAP14 + col_h(NX_R)) / 2 + col_h(NS_R) / 2
YXr = YSr + col_h(NS_R) / 2 + GAP14 + col_h(NX_R) / 2

# ─── Build diagram ─────────────────────────────────────────────────────────────

# Fixed cells (root)
lines.append('<mxCell id="0"/>')
lines.append('<mxCell id="1" parent="0"/>')

# ── Title ──────────────────────────────────────────────────────────────────────
lbl('LDNet &#8212; GLA Architecture', 130, 10, 550, 36,
    st='fontSize=18;fontStyle=1;align=center;')

# ── NN_DYN block ───────────────────────────────────────────────────────────────
rect('', XNRM - 8, DYN_T, XNRM_RD - XNRM + NW + 16, DYN_H,
     'rounded=1;fillColor=#EAF4FB;strokeColor=#6c8ebf;strokeWidth=1.5;html=1;')
lbl('&lt;b&gt;NN&lt;sub&gt;dyn&lt;/sub&gt;&lt;/b&gt; &#160; w&lt;sub&gt;dyn&lt;/sub&gt; | tanh | 7&#8211;7&#8211;d&lt;sub&gt;s&lt;/sub&gt;',
    XNRM, DYN_T + 4, 220, 18,
    st='fontSize=10;fontColor=#6c8ebf;align=left;')

norm_strip(XNRM, DYN_T + 22, DYN_H - 28)

# NN_dyn input neurons
u_col  = neuron_col(XC0, YU,  NU, 'u')
s_col_d = neuron_col(XC0, YSd, NS, 'lat')
inp_d = u_col + s_col_d

# NN_dyn hidden + output layers
h1_d = neuron_col(XC1, YDYN, NH_D, 'hid')
h2_d = neuron_col(XC2, YDYN, NH_D, 'hid')
o_d  = neuron_col(XC3D, YDYN, NO_D, 'lat')  # ds/dt → orange

weight_lines(inp_d, h1_d, step=1)
weight_lines(h1_d,  h2_d, step=1)
weight_lines(h2_d,  o_d,  step=1)

norm_strip(XNRM_RD, DYN_T + 22, DYN_H - 28)

# ds/dt and integration
lbl('&lt;b&gt;d&lt;b&gt;s&lt;/b&gt;/dt&lt;/b&gt;', XDSDT, YDYN - 12, 65, 24,
    st='fontSize=13;align=left;fontStyle=1;')
rect('&amp;#8747;dt', XINTEG, YDYN - 32, 68, 58,
     'rounded=1;html=1;strokeColor=#555555;fillColor=#f5f5f5;'
     'fontSize=18;fontStyle=1;align=center;verticalAlign=middle;')
# dashed line: ds/dt → ∫dt
arrow([(XNRM_RD + NW + 2, YDYN), (XINTEG, YDYN)],
      '#888888', w=1.5, dashed=True)

# ── s(t) column (left side, between blocks) ────────────────────────────────────
s_main = neuron_col(XS, YS, NS, 'lat')
lbl('&lt;b&gt;s&lt;/b&gt;(t)', XS - 22, YS - col_h(NS)/2 - 20, 60, 18,
    st='fontSize=12;fontStyle=1;fontColor=#f0a30a;align=center;')
lbl('&amp;isin; &amp;#8477;&lt;sup&gt;d&lt;sub&gt;s&lt;/sub&gt;&lt;/sup&gt;',
    XS - 20, YS + col_h(NS)/2 + 4, 56, 16,
    st='fontSize=10;fontColor=#BD7000;align=center;')

# ── NN_REC block ───────────────────────────────────────────────────────────────
rect('', XNRM - 8, REC_T, XNRM_RR - XNRM + NW + 16, REC_H,
     'rounded=1;fillColor=#EAF4FB;strokeColor=#6c8ebf;strokeWidth=1.5;html=1;')
lbl('&lt;b&gt;NN&lt;sub&gt;rec&lt;/sub&gt;&lt;/b&gt; &#160; w&lt;sub&gt;rec&lt;/sub&gt; | tanh | 24&#8211;24&#8211;24&#8211;24',
    XNRM, REC_T + 4, 260, 18,
    st='fontSize=10;fontColor=#6c8ebf;align=left;')

norm_strip(XNRM, REC_T + 22, REC_H - 28)

# NN_rec input neurons
s_col_r = neuron_col(XC0, YSr, NS_R, 'lat')
x_col_r = neuron_col(XC0, YXr, NX_R, 'x')
inp_r = s_col_r + x_col_r

# NN_rec hidden + output layers
h1_r = neuron_col(XC1, YREC, NH_R, 'hid')
h2_r = neuron_col(XC2, YREC, NH_R, 'hid')
h3_r = neuron_col(XC3R, YREC, NH_R, 'hid')
o_r  = neuron_col(XC4R, YREC, NO_R, 'oy')   # F_y, M_z → purple

weight_lines(inp_r, h1_r, step=1)
weight_lines(h1_r,  h2_r, step=1)
weight_lines(h2_r,  h3_r, step=1)
weight_lines(h3_r,  o_r,  step=1)

norm_strip(XNRM_RR, REC_T + 22, REC_H - 28)

# ─── Output labels ─────────────────────────────────────────────────────────────
# ỹ label + reconstruction formula
lbl('&lt;b&gt;&#7929;(&lt;b&gt;x&lt;/b&gt;,t)&lt;/b&gt;', XOUT_R, YREC - 14, 80, 28,
    st='fontSize=14;fontStyle=1;align=left;')
lbl('F&lt;sub&gt;y&lt;/sub&gt;(t),  M&lt;sub&gt;z&lt;/sub&gt;(t)', XOUT_R, YREC + 16, 120, 24,
    st='fontSize=12;fontColor=#7B3FB0;align=left;')
lbl('y&lt;sub&gt;0&lt;/sub&gt; + y&lt;sub&gt;w&lt;/sub&gt; &#8857; NN&lt;sub&gt;rec&lt;/sub&gt;',
    XOUT_R, YREC + 40, 160, 20,
    st='fontSize=10;fontColor=#999999;align=left;')

# ─── u(t) input label ──────────────────────────────────────────────────────────
lbl('&lt;b&gt;u&lt;/b&gt;(t)', 4, YU - 20, 65, 20,
    st='fontSize=13;fontStyle=1;fontColor=#446359;align=center;')
# signal names
names_u = ['W&lt;sub&gt;gust&lt;/sub&gt;', '&amp;delta;', 'h', 'h&amp;#775;', '&amp;alpha;', '&amp;alpha;&amp;#775;']
for k, (ux, uy) in enumerate(u_col):
    lbl(names_u[k], 4, uy - 8, 64, 16, st='fontSize=9;fontColor=#446359;align=right;')

# ─── x query label ─────────────────────────────────────────────────────────────
rect('&lt;b&gt;x&lt;/b&gt; &amp;isin; &amp;Omega; &amp;sub; &amp;#8477;&lt;sup&gt;d&lt;/sup&gt;',
     4, YXr - 30, 75, 56,
     'rounded=1;html=1;strokeColor=#82b366;fillColor=#d5e8d4;'
     'fontSize=11;fontStyle=1;align=center;verticalAlign=middle;')

# ─── s(0)=0 note ───────────────────────────────────────────────────────────────
lbl('s(0) = &lt;b&gt;0&lt;/b&gt;', XS - 18, DYN_T - 20, 72, 16,
    st='fontSize=10;fontColor=#999999;align=center;fontStyle=2;')

# ─── ARROWS ────────────────────────────────────────────────────────────────────
TEAL   = '#446359'
ORANGE = '#BD7000'
BLUE   = '#23445D'
PURPLE = '#7B3FB0'

# u(t) → NN_dyn input neurons (teal arrows, one per neuron)
for ux, uy in u_col:
    arrow([(68, uy), (XNRM + 1, uy)], TEAL, w=1.5)

# s(t) column → NN_dyn input (orange, goes UP from s(t) top)
S_TOP = YS - col_h(NS) / 2   # top of s(t) column
arrow([(XS, S_TOP), (XS, YSd), (XNRM + 1, YSd)], ORANGE, w=2.5)

# ∫dt → s(t) column top  (orange feedback arc)
INTEG_BOT = YDYN + 32 + 6   # bottom of ∫dt block  (≈YDYN-32+58=YDYN+26)
arrow([(XINTEG + 34, INTEG_BOT),
       (XINTEG + 34, YS - col_h(NS)/2 - 14),
       (XS, YS - col_h(NS)/2 - 14),
       (XS, YS - col_h(NS)/2)],
      ORANGE, w=2.5)

# s(t) → NN_rec input (orange, goes DOWN from s(t) bottom)
S_BOT = YS + col_h(NS) / 2
arrow([(XS, S_BOT), (XS, YSr), (XNRM + 1, YSr)], ORANGE, w=2.5)

# x query → NN_rec input (blue)
arrow([(82, YXr), (XNRM + 1, YXr)], BLUE, w=2.0)

# NN_rec output → ỹ label (purple)
arrow([(XNRM_RR + NW, YREC), (XOUT_R, YREC)], PURPLE, w=2.0)

# ─── Assemble XML ──────────────────────────────────────────────────────────────
CANVAS_W = max(XINTEG + 120, XOUT_R + 170)
CANVAS_H = REC_T + REC_H + 50

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
print(f'  Canvas: {CANVAS_W} × {CANVAS_H} px')
print(f'  Elements generated: {_id[0] - 50}')
