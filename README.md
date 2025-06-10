# ComfyUI 扭曲变换工具包 (Distortion Transformation Kit)

这是一个用于 ComfyUI 的自定义节点包，提供图像扭曲变换和掩码处理功能。

## 功能特点
![屏幕截图 2025-06-10 003941](https://github.com/user-attachments/assets/abcd14fb-1ea7-4484-9d70-f352bd3d5efd)

- **扭曲变换可视化节点**：通过交互式界面调整图像的透视变换
- **预设变换节点**：使用预定义参数直接应用透视变换
- **掩码处理工具**：包含快速和自适应的掩码补齐功能
- **四边形掩码变形**：基于掩码的图像变形工具
  
## 安装方法!


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

### 扭曲变换可视化节点 (WjbDistortionTransform![屏幕截图 2025-06-10 010541](https://github.com/user-attachments/assets/794337f8-8dcb-4754-aae1-45cd53d7bf51)
![屏幕截图 2025-06-10 010557](https://github.com/user-attachments/assets/69424904-0a74-492b-ab91-aebdbebe6ead)


这个节点提供交互式界面，让您可以直观地调整前景图像在背景上的位置和透视变换。

1. 连接背景图像和前景图像到节点
2. 执行工作流后，会弹出交互式编辑界面
3. 拖动四个角点来调整前景图像的透视变换
4. 点击"应用"按钮确认变换

### 预设变换节点 (WjbDistortionTransformDirect)
![屏幕截图 2025-06-10 011253](https://github.com/user-attachments/assets/d08bcdfc-f9e9-4b98-84b3-b3e6947daed2)

这个节点允许您通过直接设置四个角点的坐标来应用透视变换，适合需要精确控制或批处理的场景。
也可以将扭曲变换节点可视化后的八个点坐标直接连线到预设变换节点中加载相同的背景图像加载不同的前景图像后加载的前景图像的角点坐标会与扭曲变换节点中的前景图像坐标一致
### 掩码处理工具

- **补齐快速**：快速补齐不完整的掩码
- ![屏幕截图 2025-06-10 014447](https://github.com/user-attachments/assets/a42addd5-342f-48cd-a2e5-07ef9d94daae)
实战案例![屏幕截图 2025-06-10 015023](https://github.com/user-attachments/assets/21ebcf47-dd2a-42fb-b1bd-34706ea8ce20)

- **补齐慢速自适应强**：将掩码近似为多边形，支持自定义边数
- 补齐图像为四边形
![屏幕截图 2025-06-10 015425](https://github.com/user-attachments/assets/86e29e7f-0ec7-46df-9cb9-b095af0c710b)
![ComfyUI_temp_bsadg_00001_](https://github.com/user-attachments/assets/a016ea3f-9824-4bc6-90e1-f7f584ca21b9)

补齐图像为五边形
![屏幕截图 2025-06-10 015425](https://github.com/user-attachments/assets/f53391ea-d30a-472f-be53-ea239e3c8d8b)
![ComfyUI_temp_bsadg_00002_](https://github.com/user-attachments/assets/352cac6d-7b21-489d-9521-23f5dab17f54)

### 四边形掩码变形 (QuadMaskWarper)
![屏幕截图 2025-06-10 013526](https://github.com/user-attachments/assets/610d0b25-4f83-419b-bc1c-f95a6b09b692)
案例展示样机操作
![屏幕截图 2025-06-10 013441](https://github.com/user-attachments/assets/b83c14d4-6a3f-4cfc-911e-5997a334190c)

基于掩码的图像变形工具，可以将图像按照掩码形状进行变形。
详细教程 【硬核组合！ComfyUI实现PS扭曲变换、批量洗图搞定批量样机，还能优化葫芦娃相框工作流！】https://www.bilibili.com/video/BV1XNTxz8EYv?vd_source=be994b32b02aea717d97a73e61a1dd72
