# 手写字体提取与渲染工具 (Handwritten Tracing)

## 项目描述
这是一个能够将手写文字图片转换为可编辑字体文件，并支持使用生成的字体进行文本渲染的完整工具包。项目通过图像处理技术从输入的手写文字图片中提取字符轮廓，生成 SVG 格式的字符数据，然后利用 FontForge 创建自定义 TTF 字体文件，最后实现高质量的手写风格文本渲染。

## 主要功能

### 1. 字体提取 ([font_extractor.py](file:///j:/my_project/coding-net/handwritten_tracing/font_extractor.py))
- **图像预处理**：使用 OpenCV 进行灰度化、二值化处理
- **轮廓检测**：识别手写字符的边界轮廓并分离外轮廓与空洞
- **字符分组**：按字符间距自动将笔画轮廓分组合并为完整字符
- **SVG 生成**：将轮廓转换为平滑的贝塞尔曲线 SVG 路径
- **FontForge 脚本生成**：自动生成用于创建 TTF 字体的 Python 脚本

### 2. 文本渲染 ([render_text.py](file:///j:/my_project/coding-net/handwritten_tracing/render_text.py))
- **高质量渲染**：支持超采样和锐化滤镜提升清晰度
- **手写效果模拟**：通过弹性形变模拟手写笔迹的微颤效果
- **多参数控制**：支持字间距、行间距、留白、颜色等多种渲染参数
- **A4 页面布局**：可将文本按 A4 纸张规格进行批量排列输出

### 3. 主程序入口 ([main.py](file:///j:/my_project/coding-net/handwritten_tracing/main.py))
- **工作流程整合**：连接字体提取和文本渲染两个主要功能
- **批处理支持**：可对多个字符进行批量处理

## 技术特点
- **图像处理**：基于 OpenCV 和 NumPy 进行轮廓分析
- **OCR 支持**：可选集成 pytesseract 进行字符识别
- **矢量图形**：生成平滑的贝塞尔曲线 SVG 路径
- **真实手写模拟**：通过多种扰动算法模拟真实手写特征
- **高精度输出**：支持高 DPI 输出，适用于打印需求

## 性能优化
- **加速级别**：
  1. CuPy + CUDA — NVIDIA GPU，最快
  2. PyTorch CUDA — CUDA GPU 备选
  3. scipy CPU — 多线程 CPU，比 PIL 快十倍
  4. PIL 回退 — 无任何额外依赖时的最慢路径

## GPU 加速安装
如果拥有 NVIDIA GPU，可以通过以下步骤安装 GPU 加速支持：

```bash
# 查看 CUDA 版本
nvidia-smi

# 按版本安装，例如 CUDA 12.x：
pip install cupy-cuda12x
# 或者对于 CUDA 11.x：
pip install cupy-cuda11x
# 或者对于 CUDA 10.2：
pip install cupy-cuda102
```

## 渲染参数详解

### 自动字号计算公式
当使用 `font_size=None` 时，字号由以下公式自动推算：

```
font_size ∝ A4 宽度 / (repeat × 组宽 + (repeat - 1) × group_spacing + 2 × padding_x)
```

### 主要调节参数

#### 1. repeat（每行组数）- 最直接影响字号大小
| repeat | 估算字号（px） | 特点 |
|--------|----------------|------|
| 7（当前） | ≈ 46 | 密集 |
| 5 | ≈ 65 | 适中 |
| 4 | ≈ 81 | 大且稀疏 |
| 3 | ≈ 108 | 很大 |

#### 2. 次要调节参数（辅助微调）
- `group_spacing` 调小 → 字略大
- `padding_x` 调小 → 字略大（效果有限，因为已有较小的初始值）

### 推荐配置
将 `repeat` 从 7 改为 5，同时将 `rows` 从默认值改为更大的数值（如 35）以补满页高：

- **已将 repeat 改为 5，rows 改为 35**：字号会从约 46px 增大到约 65px（约大 40%），获得更舒适的大字显示效果。

## 字体生成命令
在完成字体提取过程后，会生成相应的 FontForge 脚本文件。要生成最终的 TTF 字体文件，需要运行以下命令：

```bash
fontforge.exe --script output_glyphs/generate_font.py
```

此命令将在 `output_glyphs/` 目录中查找 [generate_font.py](file://j:\my_project\coding-net\handwritten_tracing\output_glyphs\generate_font.py) 脚本并执行它，生成 [output_font.ttf](file://j:\my_project\coding-net\handwritten_tracing\output_font.ttf) 文件。

## 应用场景
- 个人手写体数字化
- 特殊字体设计
- 文档手写效果生成
- 个性化字体制作

## 技术栈
- Python
- OpenCV
- NumPy
- Pillow (PIL)
- FontForge

该项目实现了从手写图片到可用字体再到渲染输出的完整闭环，特别适合需要将手写风格转换为数字化字体的应用场景。