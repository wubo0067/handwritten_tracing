# -*- coding: utf-8 -*-
import fontforge
import os

font = fontforge.font()
font.fontname = "MyHandwrittenFont"
font.fullname = "My Handwritten Font"
font.familyname = "MyHandwrittenFont"

files = [
    (r"output_glyphs/0_控.svg", 25511),  # 控
    (r"output_glyphs/1_制.svg", 21046),  # 制
    (r"output_glyphs/2_单.svg", 21333),  # 单
    (r"output_glyphs/3_一.svg", 19968),  # 一
    (r"output_glyphs/4_变.svg", 21464),  # 变
    (r"output_glyphs/5_量.svg", 37327),  # 量
]

current_code = 0xE000
for svg_file, code in files:
    if code == -1:
        code = current_code
        current_code += 1
    glyph = font.createChar(code)
    glyph.importOutlines(svg_file)
    glyph.autoTrace()

font.generate("kongzhidanyibianliang.ttf")
print("字体已生成：kongzhidanyibianliang.ttf")
