#!/usr/bin/env python3
"""Fix green box (x label) position and reroute s(t) feedback arrow.

Changes:
  1. Edge id=460 (s(t) -> NN_rec): was a straight diagonal across the diagram;
     reroute to go BELOW (y=370) like edge id=459 already does for s(t)->NN_dyn.
  2. Green box id=417: move from x=642,y=214 to x=630,y=148  (aligned with blue neurons).
  3. Arrow id=428: update sourcePoint from (722,240) to (710,174) and
     targetPoint from (777,240) to (791,173) so it enters the left side of the blue box.
"""
import re

path = '/home/marco/LDNet_OF/figs/ldnet_arch.drawio'
with open(path, 'r', encoding='utf-8') as f:
    xml = f.read()

# ── 1. Reroute s(t)→NN_rec feedback arrow (id=460) to go below diagram ────
OLD_460 = (
    '                <mxCell id="460" value="" style="edgeStyle=none;html=1;'
    'entryX=0;entryY=0.5;entryDx=0;entryDy=0;" parent="1" source="250" target="405" edge="1">\n'
    '                    <mxGeometry relative="1" as="geometry"/>\n'
    '                </mxCell>'
)
NEW_460 = (
    '                <mxCell id="460" value="" style="edgeStyle=orthogonalEdgeStyle;html=1;'
    'exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" '
    'parent="1" source="250" target="405" edge="1">\n'
    '                    <mxGeometry relative="1" as="geometry">\n'
    '                        <Array as="points">\n'
    '                            <mxPoint x="584" y="370"/>\n'
    '                            <mxPoint x="804" y="370"/>\n'
    '                        </Array>\n'
    '                    </mxGeometry>\n'
    '                </mxCell>'
)
assert OLD_460 in xml, "id=460 old string not found!"
xml = xml.replace(OLD_460, NEW_460)

# ── 2. Move green box (id=417) from x=642,y=214 to x=630,y=148 ───────────
OLD_417 = '<mxGeometry x="642" y="214" width="80" height="52" as="geometry"/>'
NEW_417 = '<mxGeometry x="630" y="148" width="80" height="52" as="geometry"/>'
assert OLD_417 in xml, "id=417 geometry not found!"
xml = xml.replace(OLD_417, NEW_417)

# ── 3. Update x arrow (id=428): point from green box to blue neurons ───────
OLD_SRC = '<mxPoint x="722" y="240" as="sourcePoint"/>'
NEW_SRC = '<mxPoint x="710" y="174" as="sourcePoint"/>'
OLD_TGT = '<mxPoint x="777" y="240" as="targetPoint"/>'
NEW_TGT = '<mxPoint x="791" y="173" as="targetPoint"/>'
assert OLD_SRC in xml, "id=428 sourcePoint not found!"
assert OLD_TGT in xml, "id=428 targetPoint not found!"
xml = xml.replace(OLD_SRC, NEW_SRC)
xml = xml.replace(OLD_TGT, NEW_TGT)

with open(path, 'w', encoding='utf-8') as f:
    f.write(xml)

bad = set(re.findall(r'&[a-zA-Z][a-zA-Z0-9]*;', xml)) - {'&amp;', '&lt;', '&gt;', '&quot;', '&apos;'}
print('Bad XML entities:', bad or 'none')
print('Done.')
print('  s(t) arrow (id=460): rerouted below at y=370 (same style as id=459)')
print('  Green box  (id=417): moved to x=630, y=148  (center y=174 ≈ blue neurons center y=170)')
print('  x arrow    (id=428): (710,174) → (791,173)  (enters left of blue box id=503)')
