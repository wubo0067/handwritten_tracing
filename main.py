import os
import font_extractor
from render_text import render_text_image, render_a4_groups


def main():
    print("Hello from handwritten-tracing!")

    # img_path = os.path.join("pic", "Chikungunya virus.jpg")
    # font_extractor.main(img_path, char_labels=["基", "孔", "肯", "雅", "病", "毒"])

    # img_path = os.path.join("pic", "kongzhidanyibianliang.jpg")
    # font_extractor.main(img_path, char_labels=["控", "制", "单", "一", "变", "量"])

    configs = [
        {
            "text": "基孔肯雅病毒",
            "font_path": "jikongkengyabingdu.ttf",
            "output_path": "output_a4_jikong.png",
            "scale_jitter": 0.07,
        },
        {
            "text": "控制单一变量",
            "font_path": "kongzhidanyibianliang.ttf",
            "output_path": "output_a4_kongzhi.png",
            "scale_jitter": 0.07,
        },
    ]

    for config in configs:
        render_a4_groups(
            text=config["text"],
            font_path=config["font_path"],
            output_path=config["output_path"],
            repeat=6,
            rows=30,
            font_size=None,
            supersample=6,
            letter_spacing=4,
            letter_spacing_jitter=2,
            group_spacing=35,
            group_spacing_jitter=2,
            line_spacing=28,
            line_spacing_jitter=5,
            padding_x=10,
            padding_y=15,
            padding_jitter=20,
            dpi=300,
            jitter=True,
            stroke_alpha=5.0,
            stroke_sigma=1.5,
            alpha_jitter=0.7,
            sigma_jitter=0.4,
            ink_jitter=15,
            baseline_jitter=8,
            rotation_jitter=3.5,
            scale_jitter=config["scale_jitter"],
            seed=None,
        )


if __name__ == "__main__":
    main()
