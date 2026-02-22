# -*- coding: utf-8 -*-
import fontforge
import os

font = fontforge.font()
font.fontname = "MyHandwrittenFont"
font.fullname = "My Handwritten Font"
font.familyname = "MyHandwrittenFont"

files = [
    (r"output_glyphs/0_基.svg", 22522),  # 基
    (r"output_glyphs/1_孔.svg", 23380),  # 孔
    (r"output_glyphs/2_肯.svg", 32943),  # 肯
    (r"output_glyphs/3_雅.svg", 38597),  # 雅
    (r"output_glyphs/4_病.svg", 30149),  # 病
    (r"output_glyphs/5_毒.svg", 27602),  # 毒
]

current_code = 0xE000
for svg_file, code in files:
    if code == -1:
        code = current_code
        current_code += 1
    glyph = font.createChar(code)
    glyph.importOutlines(svg_file)
    glyph.autoTrace()

font.generate("jikongkengyabingdu.ttf")
print("字体已生成：jikongkengyabingdu.ttf")
