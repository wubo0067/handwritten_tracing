import os
import font_extractor
from render_text import render_text_image, render_a4_groups


def main():
    print("Hello from handwritten-tracing!")

    # img_path = os.path.join("pic", "Chikungunya virus.jpg")
    # font_extractor.main(img_path, char_labels=["基", "孔", "肯", "雅", "病", "毒"])

    render_a4_groups(
        text="基孔肯雅病毒",
        font_path="output_font.ttf",
        output_path="output_a4.png",
        repeat=7,
        rows=23,
        font_size=360,
        letter_spacing=6,  # 组内字间距基准值（px）
        letter_spacing_jitter=2,  # 字间距随机浮动范围 ±10px
        group_spacing=35,  # 组间距基准值（px）
        group_spacing_jitter=5,  # 组间距随机浮动范围 ±10px
        line_spacing=10,  # 行间距基准值（px）
        line_spacing_jitter=3,  # 行间距随机浮动范围 ±10px
        padding_x=10,
        padding_y=20,
        padding_jitter=15,  # 每行行首随机右偏移 0~15px
        dpi=150,
        jitter=True,
        stroke_alpha=4.5,  # 笔画扭曲幅度基准值
        stroke_sigma=3.0,  # 平滑半径基准值
        alpha_jitter=0.4,  # 扭曲幅度 ±40% 随机
        sigma_jitter=1.0,  # 平滑半径 ±1.0 像素随机
        ink_jitter=25,  # 墓色深浅 ±25 随机
        baseline_jitter=4,  # 字符基线 ±4px 垂直浮动
        seed=None,
    )


if __name__ == "__main__":
    main()
