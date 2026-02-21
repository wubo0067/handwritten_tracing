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
        repeat=7,  # 单行大字：3 组/行 → 字号约 457px；改为 2 可达 710px
        rows=40,
        font_size=None,
        supersample=6,  # 提升清晰度（默认 4，改 6 后边缘更平滑，内存翻 1.5 倍）
        letter_spacing=2,  # 组内字间距收窄（px）
        letter_spacing_jitter=1,
        group_spacing=30,  # 组间距大幅收窄，让组更密（px）
        group_spacing_jitter=5,
        line_spacing=15,  # 行间距收窄（px）
        line_spacing_jitter=3,
        padding_x=6,  # 左右留白收窄（px）
        padding_y=12,  # 上下留白收窄（px）
        padding_jitter=15,  # 每行行首随机右偏移 0~10px
        dpi=300,
        jitter=True,
        stroke_alpha=4.5,  # 笔画扭曲幅度基准值
        stroke_sigma=3.0,  # 平滑半径基准值
        alpha_jitter=0.6,  # 扭曲幅度 ±40% 随机
        sigma_jitter=1.0,  # 平滑半径 ±1.0 像素随机
        ink_jitter=10,  # 墨色深浅 ±25 随机
        baseline_jitter=8,  # 字符基线 ±4px 垂直浮动
        seed=None,
    )


if __name__ == "__main__":
    main()
