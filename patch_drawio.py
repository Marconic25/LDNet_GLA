#!/usr/bin/env python3
"""Patch ldnet_arch.drawio: fix colors, add (t) to signals, add missing boxes."""
import re

path = '/home/marco/LDNet_OF/figs/ldnet_arch.drawio'
with open(path, 'r', encoding='utf-8') as f:
    xml = f.read()

# ── 1. id=443: teal box around u(t) neurons (was orange) ─────────────────
xml = xml.replace(
    'id="443" value="" style="rounded=0;fillColor=none;strokeColor=#BD7000;strokeWidth=2;html=1;" vertex="1" parent="1">\n                    <mxGeometry x="155" y="72" width="26" height="123"',
    'id="443" value="" style="rounded=0;fillColor=none;strokeColor=#3d7a6e;strokeWidth=2;html=1;" vertex="1" parent="1">\n                    <mxGeometry x="155" y="72" width="26" height="147"'
)

# ── 2. Signal labels: add (t) ─────────────────────────────────────────────
xml = xml.replace(
    'id="410" value="W&lt;sub&gt;gust&lt;/sub&gt;&lt;div&gt;&lt;sub&gt;&lt;br&gt;&lt;/sub&gt;&lt;/div&gt;"',
    'id="410" value="W&lt;sub&gt;gust&lt;/sub&gt;(t)"'
)
for old, new in [
    ('<mxCell id="411" value="δ"',     '<mxCell id="411" value="δ(t)"'),
    ('<mxCell id="412" value="h"',           '<mxCell id="412" value="h(t)"'),
    ('<mxCell id="413" value="ḣ"',      '<mxCell id="413" value="ḣ(t)"'),
    ('<mxCell id="414" value="α"',      '<mxCell id="414" value="α(t)"'),
    ('<mxCell id="415" value="α̇"','<mxCell id="415" value="α̇(t)"'),
]:
    xml = xml.replace(old, new)

# ── 3. New elements ────────────────────────────────────────────────────────
NEW = (
    '\n'
    '                <!-- U_inf neuron (pale teal), gap between u(t) and s(t) groups -->\n'
    '                <mxCell id="500" value="" style="ellipse;fillColor=#a8d5c8;strokeColor=#3d7a6e;strokeWidth=1.5;html=1;" parent="1" vertex="1">\n'
    '                    <mxGeometry x="152" y="193" width="16" height="16" as="geometry"/>\n'
    '                </mxCell>\n'
    '                <!-- U_inf label -->\n'
    '                <mxCell id="501" value="U&lt;sub&gt;&#8734;&lt;/sub&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontSize=9;fontColor=#3d7a6e;align=right;fontStyle=2;" parent="1" vertex="1">\n'
    '                    <mxGeometry x="6" y="193" width="70" height="16" as="geometry"/>\n'
    '                </mxCell>\n'
    '                <!-- U_inf arrow -->\n'
    '                <mxCell id="502" value="" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#3d7a6e;strokeWidth=1.5;endArrow=block;endFill=1;" parent="1" edge="1">\n'
    '                    <mxGeometry relative="1" as="geometry">\n'
    '                        <mxPoint x="78" y="201" as="sourcePoint"/>\n'
    '                        <mxPoint x="151" y="201" as="targetPoint"/>\n'
    '                        <Array as="points"/>\n'
    '                    </mxGeometry>\n'
    '                </mxCell>\n'
    '                <!-- Blue rectangle around x spatial neurons in NN_rec (ids 261-262) -->\n'
    '                <mxCell id="503" value="" style="rounded=0;fillColor=none;strokeColor=#23445D;strokeWidth=2;html=1;" parent="1" vertex="1">\n'
    '                    <mxGeometry x="791" y="217" width="26" height="52" as="geometry"/>\n'
    '                </mxCell>\n'
    '                <!-- Purple rectangle around NN_rec output neurons (ids 281-282) -->\n'
    '                <mxCell id="504" value="" style="rounded=0;fillColor=none;strokeColor=#7B3FB0;strokeWidth=2;html=1;" parent="1" vertex="1">\n'
    '                    <mxGeometry x="999" y="165" width="26" height="46" as="geometry"/>\n'
    '                </mxCell>\n'
)
xml = xml.replace('            </root>', NEW + '            </root>')

# ── 4. Canvas size ────────────────────────────────────────────────────────
xml = re.sub(r'pageWidth="\d+"',  'pageWidth="1230"',  xml)
xml = re.sub(r'pageHeight="\d+"', 'pageHeight="400"', xml)

with open(path, 'w', encoding='utf-8') as f:
    f.write(xml)

bad = set(re.findall(r'&[a-zA-Z][a-zA-Z0-9]*;', xml)) - {'&amp;', '&lt;', '&gt;', '&quot;', '&apos;'}
print('Bad XML entities:', bad or 'none')
print('Done.')
