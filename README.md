# ComfyUI 扭曲变换工具包 (Distortion Transformation Kit)

这是一个用于 ComfyUI 的自定义节点包，提供图像扭曲变换和掩码处理功能。

## 功能特点

- **扭曲变换节点**：通过交互式界面调整图像的透视变换
- **预设变换节点**：使用预定义参数直接应用透视变换
- **掩码处理工具**：包含快速和自适应的掩码补齐功能
- **四边形掩码变形**：基于掩码的图像变形工具

## 安装方法

### 方法一：使用 ComfyUI Manager

1. 在 ComfyUI 中安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 在 Manager 界面中搜索 "Distortion Transformation Kit" 并安装

### 方法二：手动安装

1. 将此仓库克隆到您的 ComfyUI 的 `custom_nodes` 目录：
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/YOUR-USERNAME/comfyui-Distortion-Transformation-Kit.git
```

2. 安装依赖：
```bash
cd comfyui-Distortion-Transformation-Kit
pip install -r requirements.txt
```

3. 重启 ComfyUI

## 使用说明

### 扭曲变换节点 (WjbDistortionTransform)

这个节点提供交互式界面，让您可以直观地调整前景图像在背景上的位置和透视变换。

1. 连接背景图像和前景图像到节点
2. 执行工作流后，会弹出交互式编辑界面
3. 拖动四个角点来调整前景图像的透视变换
4. 点击"应用"按钮确认变换

### 预设变换节点 (WjbDistortionTransformDirect)

这个节点允许您通过直接设置四个角点的坐标来应用透视变换，适合需要精确控制或批处理的场景。

### 掩码处理工具

- **补齐快速**：快速补齐不完整的掩码
- **补齐慢速自适应强**：将掩码近似为多边形，支持自定义边数

### 四边形掩码变形 (QuadMaskWarper)

基于掩码的图像变形工具，可以将图像按照掩码形状进行变形。
