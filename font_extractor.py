import cv2
import numpy as np
import os
import sys

# 尝试导入 pytesseract 用于字符识别
try:
    import pytesseract

    HAS_OCR = True
except ImportError:
    HAS_OCR = False


def _smooth_contour_to_path(points):
    """
    将轮廓点列表转换为平滑三次贝塞尔曲线 SVG 路径字符串（Catmull-Rom 转换）。
    """
    n = len(points)
    if n < 3:
        return ""

    p = points
    parts = [f"M {p[0][0]:.2f} {p[0][1]:.2f} "]

    for i in range(n):
        p1 = p[i]
        p2 = p[(i + 1) % n]
        p0 = p[(i - 1) % n]
        p3 = p[(i + 2) % n]

        # Catmull-Rom -> Cubic Bezier 控制点（张力系数 1/6）
        cp1x = p1[0] + (p2[0] - p0[0]) / 6.0
        cp1y = p1[1] + (p2[1] - p0[1]) / 6.0
        cp2x = p2[0] - (p3[0] - p1[0]) / 6.0
        cp2y = p2[1] - (p3[1] - p1[1]) / 6.0

        parts.append(
            f"C {cp1x:.2f} {cp1y:.2f} {cp2x:.2f} {cp2y:.2f} "
            f"{p2[0]:.2f} {p2[1]:.2f} "
        )

    parts.append("Z ")
    return "".join(parts)


def create_svg(contours, width, height, filename):
    """将 OpenCV 轮廓转换为平滑贝塞尔曲线 SVG 路径并保存。
    使用 fill-rule=evenodd，外轮廓填黑，空洞轮廓自动镂空。
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        )
        f.write('<path fill-rule="evenodd" d="')

        for cnt in contours:
            # 用较小的 epsilon 轻度简化，去掉冗余共线点但保留笔画细节
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) < 3:
                continue

            pts = [pt[0].tolist() for pt in approx]
            f.write(_smooth_contour_to_path(pts))

        f.write('" fill="black" stroke="none"/>')
        f.write("</svg>")


def group_contours_by_character(contours, img_width, expected_count):
    """
    将分散的笔画轮廓按字符分组。
    使用"最大间距切割法"：找到相邻轮廓间最大的 N-1 个水平空隙作为分界线，
    比等宽区间划分更能适应字符间距不均匀的情况。
    """
    if not contours:
        return []

    # 计算每个轮廓的 bbox 及左右边界
    bbox_list = [(cv2.boundingRect(c), c) for c in contours]
    # 按 bbox 左边界 X 排序
    bbox_list.sort(key=lambda b: b[0][0])

    sorted_cnts = [item[1] for item in bbox_list]
    sorted_boxes = [item[0] for item in bbox_list]  # (x, y, w, h)

    # 计算相邻轮廓之间的水平间隙：下一个轮廓的左边界 - 当前轮廓的右边界
    gaps = []
    for i in range(len(sorted_boxes) - 1):
        cur_right = sorted_boxes[i][0] + sorted_boxes[i][2]
        next_left = sorted_boxes[i + 1][0]
        gap = next_left - cur_right
        gaps.append((gap, i))  # (间距大小, 当前轮廓索引)

    # 取最大的 (expected_count - 1) 个间隙作为切割点，并按位置排序
    n_cuts = expected_count - 1
    if len(gaps) >= n_cuts:
        cut_indices = sorted(
            [idx for _, idx in sorted(gaps, key=lambda g: -g[0])[:n_cuts]]
        )
    else:
        # 间隙数量不足时退回等宽划分
        cut_indices = []
        x_min = sorted_boxes[0][0]
        x_max = sorted_boxes[-1][0] + sorted_boxes[-1][2]
        interval = (x_max - x_min) / expected_count
        groups = [[] for _ in range(expected_count)]
        for cnt, box in zip(sorted_cnts, sorted_boxes):
            cx = box[0] + box[2] / 2
            group_idx = min(int((cx - x_min) / interval), expected_count - 1)
            groups[group_idx].append(cnt)
        return [g for g in groups if g]

    # 按切割点分组
    groups = []
    start = 0
    for cut in cut_indices:
        groups.append(sorted_cnts[start : cut + 1])
        start = cut + 1
    groups.append(sorted_cnts[start:])

    # 过滤空组
    groups = [g for g in groups if g]
    return groups


def get_group_bounding_box(contours):
    """获取一组轮廓的整体边界框。"""
    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
    return x_min, y_min, x_max - x_min, y_max - y_min


def main(image_path, output_dir="output_glyphs", char_labels=None, expected_count=None):
    """
    主函数。
    :param image_path: 图片路径
    :param output_dir: 输出目录
    :param char_labels: 手动指定的字符列表，如 ['基','孔','肯','雅','病','毒']
    :param expected_count: 预期字符数量（当 char_labels 未指定时使用）
    """
    # 修复：避免局部变量遮蔽全局 HAS_OCR
    _has_ocr = HAS_OCR

    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在处理图片：{image_path}")

    # 1. 读取图片并预处理
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 查找轮廓（RETR_CCOMP 获取两层层级：外轮廓 + 空洞）
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    print(f"找到 {len(contours)} 个潜在轮廓。")

    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

    # 分离外轮廓（parent == -1）与空洞轮廓（parent >= 0）
    # 空洞 key 为其父外轮廓在 contours 中的索引
    hole_by_parent: dict[int, list] = {}
    external_indices = []
    for i in range(len(contours)):
        parent = hierarchy[i][3]
        if parent == -1:
            external_indices.append(i)
            hole_by_parent[i] = []
        else:
            # 只收集一级空洞（直接父级是外轮廓）
            if parent in hole_by_parent:
                hole_by_parent[parent].append(contours[i])

    # 过滤噪点（只过滤外轮廓；空洞跟随父轮廓，不单独过滤）
    min_area = 50
    valid_ext_indices = [
        i for i in external_indices if cv2.contourArea(contours[i]) > min_area
    ]
    valid_contours = [contours[i] for i in valid_ext_indices]

    # 建立「外轮廓对象 id → 空洞列表」映射，供后续分组后查找
    hole_by_id = {id(contours[i]): hole_by_parent[i] for i in valid_ext_indices}

    print(f"保留 {len(valid_contours)} 个有效外轮廓。")

    # 3. 确定分组数量和字符标签
    if char_labels:
        n_chars = len(char_labels)
        print(f"使用手动指定的字符标签：{''.join(char_labels)}")
    elif expected_count:
        n_chars = expected_count
        char_labels = [f"char_{i}" for i in range(n_chars)]
    else:
        n_chars = len(valid_contours)
        char_labels = [f"char_{i}" for i in range(n_chars)]

    # 4. 将轮廓按字符分组（合并同一字的笔画）
    if n_chars < len(valid_contours):
        print(f"将 {len(valid_contours)} 个轮廓合并为 {n_chars} 个字符组...")
        char_groups = group_contours_by_character(valid_contours, img.shape[1], n_chars)
    else:
        # 每个轮廓就是一个字符
        char_groups = [[c] for c in valid_contours]

    # 按 X 坐标对分组排序（从左到右）
    char_groups.sort(key=lambda g: min(cv2.boundingRect(c)[0] for c in g))

    print(f"最终分为 {len(char_groups)} 个字符组。")

    # 5. 提取并保存为 SVG
    generated_files = []
    pad = 10

    for i, group in enumerate(char_groups):
        # 获取该组的整体边界框
        x, y, w, h = get_group_bounding_box(group)

        # 提取字符 ROI
        roi = thresh[
            max(0, y - pad) : min(thresh.shape[0], y + h + pad),
            max(0, x - pad) : min(thresh.shape[1], x + w + pad),
        ]

        # 确定字符名称
        if i < len(char_labels):
            char_name = char_labels[i]
        else:
            char_name = f"char_{i}"

        # 保存 ROI 为 PNG 预览（可选）
        roi_filename = os.path.join(output_dir, f"{i}_{char_name}_preview.png")
        roi_visible = cv2.bitwise_not(roi)  # 黑字白底，方便查看
        cv2.imwrite(roi_filename, roi_visible)

        # 将轮廓坐标平移到 ROI 坐标系（外轮廓 + 对应空洞）
        cnt_shifted_list = []
        for cnt in group:
            cnt_shifted = cnt - [x - pad, y - pad]
            cnt_shifted_list.append(cnt_shifted)
            # 附加该外轮廓的空洞，evenodd 规则会将其渲染为镂空
            for hole in hole_by_id.get(id(cnt), []):
                hole_shifted = hole - [x - pad, y - pad]
                cnt_shifted_list.append(hole_shifted)

        svg_filename = os.path.join(output_dir, f"{i}_{char_name}.svg")
        create_svg(cnt_shifted_list, w + 2 * pad, h + 2 * pad, svg_filename)

        generated_files.append(
            {"path": svg_filename, "name": char_name, "char": char_name}
        )
        print(f"已生成：{svg_filename}  (包含 {len(group)} 个笔画轮廓)")

    # 6. 生成 FontForge 脚本
    ff_script_path = os.path.join(output_dir, "generate_font.py")
    with open(ff_script_path, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write("import fontforge\n")
        f.write("import os\n\n")
        f.write("font = fontforge.font()\n")
        f.write("font.fontname = 'MyHandwrittenFont'\n")
        f.write("font.fullname = 'My Handwritten Font'\n")
        f.write("font.familyname = 'MyHandwrittenFont'\n\n")
        f.write("files = [\n")
        for item in generated_files:
            char = item["char"]
            svg_path = item["path"].replace(os.sep, "/")
            if len(char) == 1:
                char_code = ord(char)
                f.write(f"    (r'{svg_path}', {char_code}),  # {char}\n")
            else:
                f.write(f"    (r'{svg_path}', -1),  # 未知：{char}\n")
        f.write("]\n\n")
        f.write("current_code = 0xE000\n")
        f.write("for svg_file, code in files:\n")
        f.write("    if code == -1:\n")
        f.write("        code = current_code\n")
        f.write("        current_code += 1\n")
        f.write("    glyph = font.createChar(code)\n")
        f.write("    glyph.importOutlines(svg_file)\n")
        f.write("    glyph.autoTrace()\n\n")
        f.write("font.generate('output_font.ttf')\n")
        f.write("print('字体已生成：output_font.ttf')\n")

    print(f"\n完成！已提取 {len(generated_files)} 个字符。")
    print(f"运行以下命令生成 TTF 字体文件：")
    print(f"  fontforge -script {ff_script_path}")
