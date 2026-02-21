"""
使用生成的手写字体渲染文字并输出为图片。
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import math
import numpy as np


def _elastic_distort(
    img_rgba: Image.Image, alpha: float, sigma: float, rng: random.Random
) -> Image.Image:
    """
    对 RGBA 字符图像施加弹性形变，模拟手写笔迹微颤抚动。

    原理：生成两张高斯平滑的随机噪声场作为 dx/dy 位移层，
    每个像素按目标坐标采样原图，就像笔画边缘在微颤抚动。

    :param img_rgba: 要扭曲的 RGBA 图像（字符小画布）
    :param alpha:    位移幅度（像素），越大笔画边缘抖动越明显，建议 1~4
    :param sigma:    抑制平滑半径，越大形变越平滑连贯（波浪感），建议 2~6
    :param rng:      random.Random 实例，保证可复现
    :return:         扭曲后的 RGBA 图像（尺寸不变）
    """
    arr = np.array(img_rgba, dtype=np.float32)  # (H, W, 4)
    h, w = arr.shape[:2]

    def _make_smooth_field() -> np.ndarray:
        noise = np.array(
            [[rng.gauss(0, 1) for _ in range(w)] for _ in range(h)],
            dtype=np.float32,
        )
        lo, hi = noise.min(), noise.max()
        if hi - lo < 1e-6:
            return noise
        norm = ((noise - lo) / (hi - lo) * 255).astype(np.uint8)
        blurred = Image.fromarray(norm, mode="L").filter(
            ImageFilter.GaussianBlur(sigma)
        )
        result = np.array(blurred, dtype=np.float32) / 255.0 * 2 - 1  # [-1, 1]
        return result * alpha

    dx = _make_smooth_field()  # (H, W)
    dy = _make_smooth_field()

    gy, gx = np.mgrid[0:h, 0:w]
    src_x = np.clip(gx + dx, 0, w - 1).astype(np.int32)
    src_y = np.clip(gy + dy, 0, h - 1).astype(np.int32)

    warped = arr[src_y, src_x]  # (H, W, 4)
    return Image.fromarray(warped.astype(np.uint8), mode="RGBA")


def _render_char_with_stroke_jitter(
    char: str,
    font,
    text_color: tuple,
    alpha: float,
    sigma: float,
    rng: random.Random,
) -> Image.Image:
    """
    将单字符渲染为 RGBA 小画布并施加笔画弹性形变。

    :param char:       字符
    :param font:       PIL ImageFont 对象
    :param text_color: 文字颜色 (R, G, B)，可在外部按字随机调整以模拟墨色深浅
    :param alpha:      笔画扭曲幅度（像素），每字独立传入以形成差异
    :param sigma:      平滑半径，每字独立传入以形成差异
    :param rng:        random.Random 实例
    :return:           RGBA Image
    """
    dummy = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), char, font=font)
    cw, ch = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 为形变留边距，防止笔画被裁剪
    margin = int(alpha * 2 + sigma * 2)
    img = Image.new("RGBA", (cw + margin * 2, ch + margin * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (margin - bbox[0], margin - bbox[1]), char, font=font, fill=(*text_color, 255)
    )

    return _elastic_distort(img, alpha, sigma, rng)


def render_text_image(
    text: str,
    font_path: str,
    output_path: str,
    font_size: int = 120,
    padding: int = 40,
    letter_spacing: int = 20,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    supersample: int = 4,
):
    """
    使用指定字体将文字渲染为图片，支持字间距，通过超采样 + 锐化提升清晰度。

    :param text:           要渲染的文字字符串
    :param font_path:      TTF 字体文件路径
    :param output_path:    输出图片路径（PNG/JPEG 等）
    :param font_size:      最终输出图片中的字号（像素），默认 120
    :param padding:        图片四周留白（像素），默认 40
    :param letter_spacing: 字与字之间的间距（像素），默认 20
    :param bg_color:       背景颜色 RGB 元组，默认白色 (255, 255, 255)
    :param text_color:     文字颜色 RGB 元组，默认黑色 (0, 0, 0)
    :param supersample:    超采样倍数，内部以此倍数放大绘制后再缩小，
                           值越大抗锯齿效果越好，但内存占用也越多，建议 2~4
    """
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"找不到字体文件：{font_path}")

    s = supersample
    font = ImageFont.truetype(font_path, font_size * s)
    pad_s = padding * s
    spacing_s = letter_spacing * s

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    char_bboxes = [draw.textbbox((0, 0), ch, font=font) for ch in text]
    char_widths = [b[2] - b[0] for b in char_bboxes]
    char_heights = [b[3] - b[1] for b in char_bboxes]

    total_w = sum(char_widths) + spacing_s * (len(text) - 1)
    max_h = max(char_heights)

    hi_res = Image.new("RGB", (total_w + pad_s * 2, max_h + pad_s * 2), color=bg_color)
    draw = ImageDraw.Draw(hi_res)

    x_cursor = pad_s
    for ch, bbox, cw, ch_h in zip(text, char_bboxes, char_widths, char_heights):
        y = pad_s + (max_h - ch_h) // 2 - bbox[1]
        draw.text((x_cursor - bbox[0], y), ch, font=font, fill=text_color)
        x_cursor += cw + spacing_s

    final_w = (total_w + pad_s * 2) // s
    final_h = (max_h + pad_s * 2) // s
    img = hi_res.resize((final_w, final_h), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))
    img.save(output_path, dpi=(150, 150))
    print(f"图片已保存：{output_path}  ({final_w}x{final_h} px)")


def render_a4_groups(
    text: str,
    font_path: str,
    output_path: str,
    repeat: int = 5,
    rows: int = 1,
    letter_spacing: int = 15,
    letter_spacing_jitter: int = 5,
    group_spacing: int = 80,
    group_spacing_jitter: int = 10,
    line_spacing: int = 40,
    line_spacing_jitter: int = 8,
    padding_x: int = 80,
    padding_y: int = 80,
    padding_jitter: int = 30,
    font_size: int | None = None,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    dpi: int = 150,
    supersample: int = 4,
    jitter: bool = True,
    stroke_alpha: float = 2.5,
    stroke_sigma: float = 3.0,
    alpha_jitter: float = 0.4,
    sigma_jitter: float = 1.0,
    ink_jitter: int = 25,
    baseline_jitter: int = 4,
    seed: int | None = None,
):
    """
    在 A4 竖版页面上将 text 重复 repeat 次排成一行。

    :param text:           每组要渲染的文字，如 "基孔肯雅病毒"
    :param font_path:      TTF 字体文件路径
    :param output_path:    输出图片路径（建议 .png）
    :param repeat:         重复组数（每行），默认 5
    :param rows:           行数，默认 1
    :param letter_spacing:        同一组内相邻字之间的基准间距（像素），默认 15
    :param letter_spacing_jitter: 字间距随机浮动范围（像素），实际间距在
                                  [letter_spacing - jitter, letter_spacing + jitter] 内随机，默认 5
    :param group_spacing:         相邻两组之间的基准间距（像素），应大于 letter_spacing，默认 60
    :param group_spacing_jitter:  组间距随机浮动范围（像素），实际间距在
                                  [group_spacing - jitter, group_spacing + jitter] 内随机，默认 10
    :param line_spacing:          相邻行之间的基准间距（像素，上行底部到下行顶部），默认 40
    :param line_spacing_jitter:   行间距随机浮动范围（像素），实际间距在
                                  [line_spacing - jitter, line_spacing + jitter] 内随机，默认 8
    :param padding_x:      左右留白基准值（像素），默认 80
    :param padding_y:      上下留白基准值（像素），默认 80
    :param padding_jitter: 每行行首随机右偏移范围（像素），在 [0, padding_jitter]
                           内随机，使各行行首不整齐对齐，默认 30
    :param font_size:      字号（像素）。传入具体数字时以该字号固定渲染，
                           不传（默认 None）时自动计算字号以恰好铺满页面宽度。
                           通过调小字号可将更多组排入同一行。
    :param bg_color:       背景颜色 RGB 元组，默认白色 (255, 255, 255)
    :param text_color:     文字颜色 RGB 元组，默认黑色 (0, 0, 0)
    :param dpi:            输出图片的 DPI，影响物理尺寸精度；
                           150 DPI 时 A4 竖版约为 1240×1754 px，默认 150
    :param jitter:          是否开启笔画微颤抚动以模拟手写笔迹，默认 True
    :param stroke_alpha:    笔画扭曲幅度基准值（像素），越大笔画边缘抖动越明显，建议 1~5，默认 2.5
    :param stroke_sigma:    形变平滑半径基准值，越大抖动越平滑连贯，建议 2~6，默认 3.0
    :param alpha_jitter:    扭曲幅度每字随机倍率范围，实际 alpha 在
                            [stroke_alpha*(1-j), stroke_alpha*(1+j)] 内随机，默认 0.4
    :param sigma_jitter:    平滑半径每字随机偏移范围（像素），默认 1.0
    :param ink_jitter:      毎字墓色深浅随机变动范围（RGB 分量偏移像素值），
                            模拟笔压轻重导致的淡淡深深，默认 25
    :param baseline_jitter: 每字基线随机垂直偏移范围（像素），模拟字符上下浮动，默认 4
    :param seed:            随机种子，传入整数可固定扰动结果以便复现，默认 None
    """
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"找不到字体文件：{font_path}")

    rng = random.Random(seed)

    # A4 竖版尺寸（mm → px）
    a4_w_px = int(210 / 25.4 * dpi)  # ≈ 1240
    a4_h_px = int(297 / 25.4 * dpi)  # ≈ 1754

    s = supersample
    pad_x_s = padding_x * s
    pad_y_s = padding_y * s
    ls_s = letter_spacing * s
    gs_s = group_spacing * s

    # 步骤 1：用参考字号 100*s 测量各字符宽高
    ref_font = ImageFont.truetype(font_path, 100 * s)
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    ref_bboxes = [draw.textbbox((0, 0), ch, font=ref_font) for ch in text]
    ref_group_w = sum(b[2] - b[0] for b in ref_bboxes) + ls_s * (len(text) - 1)

    # 步骤 2：按目标页宽反推字号缩放比（仅在未手动指定字号时执行）
    ref_total_w = repeat * ref_group_w + (repeat - 1) * gs_s + 2 * pad_x_s
    if font_size is None:
        scale = (a4_w_px * s) / ref_total_w
        font_size = max(1, int(100 * scale))

    font = ImageFont.truetype(font_path, font_size)
    bboxes = [draw.textbbox((0, 0), ch, font=font) for ch in text]
    widths = [b[2] - b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    group_w = sum(widths) + ls_s * (len(text) - 1)
    max_h = max(heights)

    # 步骤 3：高分辨率绘制
    # 预先计算各行的 y 起始偏移（行间距独立抖动）
    row_y_offsets = [pad_y_s]
    for r in range(1, rows):
        js = line_spacing * s + rng.randint(
            -line_spacing_jitter * s, line_spacing_jitter * s
        )
        row_y_offsets.append(row_y_offsets[-1] + max_h + js)

    # canvas_w 额外加上 padding_jitter 的最大偏移量，防止偏移后内容溢出右侧
    canvas_w = repeat * group_w + (repeat - 1) * gs_s + 2 * pad_x_s + padding_jitter * s
    canvas_h = row_y_offsets[-1] + max_h + pad_y_s

    hi_res = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)

    for row_idx in range(rows):
        y_row = row_y_offsets[row_idx]
        row_x_offset = rng.randint(0, padding_jitter * s)
        x_cursor = pad_x_s + row_x_offset
        for g in range(repeat):
            for i, (ch, bbox, cw, ch_h) in enumerate(
                zip(text, bboxes, widths, heights)
            ):
                y_base = y_row + (max_h - ch_h) // 2

                if jitter:
                    # 每字独立生成四个随机因子，使相同字的不同副本形态各不相同
                    char_alpha = (
                        stroke_alpha
                        * s
                        * rng.uniform(1 - alpha_jitter, 1 + alpha_jitter)
                    )
                    char_sigma = max(
                        0.5 * s,
                        stroke_sigma * s
                        + rng.uniform(-sigma_jitter * s, sigma_jitter * s),
                    )
                    ink_delta = rng.randint(-ink_jitter, ink_jitter)
                    char_color = tuple(
                        max(0, min(255, c + ink_delta)) for c in text_color
                    )
                    y_offset = rng.randint(-baseline_jitter * s, baseline_jitter * s)

                    glyph = _render_char_with_stroke_jitter(
                        ch, font, char_color, char_alpha, char_sigma, rng
                    )
                    # 小图中心对齐到字符理论位置，并加入基线偏移
                    margin = (glyph.width - cw) // 2
                    px = x_cursor - bbox[0] - margin
                    py = y_base - bbox[1] - margin + y_offset
                    hi_res.paste(glyph, (px, py), mask=glyph)
                else:
                    draw = ImageDraw.Draw(hi_res)
                    draw.text(
                        (x_cursor - bbox[0], y_base - bbox[1]),
                        ch,
                        font=font,
                        fill=text_color,
                    )

                # 字间距加随机抖动（最后一个字不加间距）
                if i < len(text) - 1:
                    jittered_ls = ls_s + rng.randint(
                        -letter_spacing_jitter * s, letter_spacing_jitter * s
                    )
                    x_cursor += cw + jittered_ls
                else:
                    x_cursor += cw

            # 组间距加随机抖动
            jittered_gs = gs_s + rng.randint(
                -group_spacing_jitter * s, group_spacing_jitter * s
            )
            x_cursor += jittered_gs

    # 步骤 4：缩小到 A4 尺寸
    final_w = canvas_w // s
    final_h = canvas_h // s
    if final_h > a4_h_px:
        ratio = a4_h_px / final_h
        final_w = int(final_w * ratio)
        final_h = a4_h_px

    img = hi_res.resize((final_w, final_h), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))

    # 步骤 5：居中贴到 A4 白底画布
    a4 = Image.new("RGB", (a4_w_px, a4_h_px), color=bg_color)
    a4.paste(img, ((a4_w_px - final_w) // 2, (a4_h_px - final_h) // 2))
    a4.save(output_path, dpi=(dpi, dpi))
    print(
        f"A4 图片已保存：{output_path}  "
        f"({a4_w_px}x{a4_h_px} px @ {dpi} DPI，字号≈{font_size // s}px)"
    )
