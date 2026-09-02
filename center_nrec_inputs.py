#!/usr/bin/env python3
"""Shift NN_rec input column (s(t), x, u(t)) up by 70px to center on hidden layer."""
import re

path = '/home/marco/LDNet_OF/figs/ldnet_arch.drawio'
with open(path, 'r', encoding='utf-8') as f:
    xml = f.read()

SHIFT = 70  # move up by 70px → centers input group on h1 (center y=188)

# 1. Neuron geometries at x=796 (s(t), x, u(t) in NN_rec)
for y in [128, 148, 168, 188, 222, 242, 276, 296, 316, 336, 356, 376]:
    xml = xml.replace(
        f'<mxGeometry x="796" y="{y}" width="16" height="16" as="geometry"/>',
        f'<mxGeometry x="796" y="{y - SHIFT}" width="16" height="16" as="geometry"/>'
    )

# 2. Box geometries at x=791
xml = xml.replace(
    '<mxGeometry x="791" y="123" width="26" height="86" as="geometry"/>',
    '<mxGeometry x="791" y="53" width="26" height="86" as="geometry"/>'
)
xml = xml.replace(
    '<mxGeometry x="791" y="217" width="26" height="52" as="geometry"/>',
    '<mxGeometry x="791" y="147" width="26" height="52" as="geometry"/>'
)
xml = xml.replace(
    '<mxGeometry x="791" y="271" width="26" height="121" as="geometry"/>',
    '<mxGeometry x="791" y="201" width="26" height="121" as="geometry"/>'
)

# 3. Weight line sourcePoints at x=804 (input neurons → h1)
for y in [136, 156, 176, 196, 230, 250, 284, 304, 324, 344, 364, 384]:
    xml = xml.replace(
        f'<mxPoint x="804" y="{y}" as="sourcePoint"/>',
        f'<mxPoint x="804" y="{y - SHIFT}" as="sourcePoint"/>'
    )

# 4. u(t) label in NN_rec
xml = xml.replace(
    '<mxGeometry x="700" y="323" width="84" height="18" as="geometry"/>',
    '<mxGeometry x="700" y="253" width="84" height="18" as="geometry"/>'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(xml)

bad = set(re.findall(r'&[a-zA-Z][a-zA-Z0-9]*;', xml)) - {'&amp;', '&lt;', '&gt;', '&quot;', '&apos;'}
print('Bad XML entities:', bad or 'none')
print('Done. Input group centered: top y=53, bottom y=322, center=187.5 ≈ h1 center=188')
