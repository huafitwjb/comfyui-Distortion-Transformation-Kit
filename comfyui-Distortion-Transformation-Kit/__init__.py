import torch
import numpy as np
from server import PromptServer
from threading import Event
from aiohttp import web
import json
import base64
import io
from PIL import Image, ImageDraw
import traceback
import cv2
import itertools
import sympy
from comfy import model_management
import comfy.utils

# 使用字典存储节点状态
node_data = {}

class WjbDistortionTransform:
    """wjb扭曲变换节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "背景图像": ("IMAGE",),
                "前景图像": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("混合图像", "左上X", "左上Y", "右上X", "右上Y", "右下X", "右下Y", "左下X", "左下Y")
    FUNCTION = "blend_images"
    CATEGORY = "Distortion Transformation Kit"
    OUTPUT_NODE = True

    def blend_images(self, 背景图像, 前景图像, unique_id):
        try:
            node_id = unique_id
            event = Event()
            node_data[node_id] = {
                "event": event,
                "result": None,
                "background": 背景图像,
                "foreground": 前景图像,
                "corner_points": None  # 添加用于存储角点坐标的字段
            }
            
            # 准备背景预览图像
            background_preview = (torch.clamp(背景图像.clone(), 0, 1) * 255).cpu().numpy().astype(np.uint8)[0]
            bg_pil_image = Image.fromarray(background_preview)
            bg_buffer = io.BytesIO()
            bg_pil_image.save(bg_buffer, format="PNG")
            bg_base64_image = base64.b64encode(bg_buffer.getvalue()).decode('utf-8')
            
            # 准备前景预览图像
            foreground_preview = (torch.clamp(前景图像.clone(), 0, 1) * 255).cpu().numpy().astype(np.uint8)[0]
            fg_pil_image = Image.fromarray(foreground_preview)
            fg_buffer = io.BytesIO()
            fg_pil_image.save(fg_buffer, format="PNG")
            fg_base64_image = base64.b64encode(fg_buffer.getvalue()).decode('utf-8')
            
            try:
                PromptServer.instance.send_sync("popup_blender_update", {
                    "node_id": node_id,
                    "background_image": f"data:image/png;base64,{bg_base64_image}",
                    "foreground_image": f"data:image/png;base64,{fg_base64_image}"
                })
                
                if not event.wait(timeout=300):  # 5分钟超时
                    if node_id in node_data:
                        del node_data[node_id]
                    return (背景图像, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                result_image = node_data[node_id]["result"]
                corner_points = node_data[node_id]["corner_points"] or []
                del node_data[node_id]
                
                # 提取角点坐标，如果不存在则使用默认值
                if len(corner_points) == 4:
                    top_left_x = float(corner_points[0]["x"])
                    top_left_y = float(corner_points[0]["y"])
                    top_right_x = float(corner_points[1]["x"])
                    top_right_y = float(corner_points[1]["y"])
                    bottom_right_x = float(corner_points[2]["x"])
                    bottom_right_y = float(corner_points[2]["y"])
                    bottom_left_x = float(corner_points[3]["x"])
                    bottom_left_y = float(corner_points[3]["y"])
                else:
                    top_left_x = top_left_y = top_right_x = top_right_y = bottom_right_x = bottom_right_y = bottom_left_x = bottom_left_y = 0.0
                
                return (
                    result_image if result_image is not None else 背景图像,
                    top_left_x, top_left_y,
                    top_right_x, top_right_y,
                    bottom_right_x, bottom_right_y,
                    bottom_left_x, bottom_left_y
                )
                
            except Exception as e:
                print(f"扭曲变换错误: {str(e)}")
                traceback.print_exc()
                if node_id in node_data:
                    del node_data[node_id]
                return (背景图像, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        except Exception as e:
            print(f"扭曲变换错误: {str(e)}")
            traceback.print_exc()
            if node_id in node_data:
                del node_data[node_id]
            return (背景图像, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

class WjbDistortionTransformDirect:
    """预设变换节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "背景图像": ("IMAGE",),
                "前景图像": ("IMAGE",),
                "左上X": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "左上Y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "右上X": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "右上Y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "右下X": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "右下Y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "左下X": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "左下Y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("混合图像",)
    FUNCTION = "apply_transform"
    CATEGORY = "Distortion Transformation Kit"
    OUTPUT_NODE = True

    def apply_transform(self, 背景图像, 前景图像, 左上X, 左上Y, 右上X, 右上Y, 右下X, 右下Y, 左下X, 左下Y):
        try:
            # 获取图像尺寸
            bg_height, bg_width = 背景图像.shape[1:3]
            fg_height, fg_width = 前景图像.shape[1:3]
            
            # 转换为PIL图像处理
            bg_pil = Image.fromarray((torch.clamp(背景图像.clone(), 0, 1) * 255).cpu().numpy().astype(np.uint8)[0])
            fg_pil = Image.fromarray((torch.clamp(前景图像.clone(), 0, 1) * 255).cpu().numpy().astype(np.uint8)[0])
            
            # 创建结果图像（复制背景）
            result_pil = bg_pil.copy()
            
            # 检查坐标是否有效（至少需要组成一个四边形）
            if 左上X == 0 and 左上Y == 0 and 右上X == 0 and 右上Y == 0 and 右下X == 0 and 右下Y == 0 and 左下X == 0 and 左下Y == 0:
                # 如果坐标都是0，则居中放置前景图像
                fg_scale = min(bg_width * 0.8 / fg_width, bg_height * 0.8 / fg_height)
                fg_w, fg_h = int(fg_width * fg_scale), int(fg_height * fg_scale)
                fg_x, fg_y = (bg_width - fg_w) // 2, (bg_height - fg_h) // 2
                
                # 调整大小并粘贴
                fg_resized = fg_pil.resize((fg_w, fg_h), Image.LANCZOS)
                result_pil.paste(fg_resized, (fg_x, fg_y), fg_resized if fg_resized.mode == 'RGBA' else None)
            else:
                # 设置目标四边形的坐标
                dst_quad = [
                    (左上X, 左上Y),
                    (右上X, 右上Y),
                    (右下X, 右下Y),
                    (左下X, 左下Y)
                ]
                
                # 源四边形（前景图像的原始矩形）
                fg_scale = min(bg_width * 0.8 / fg_width, bg_height * 0.8 / fg_height)
                fg_w, fg_h = int(fg_width * fg_scale), int(fg_height * fg_scale)
                fg_x, fg_y = (bg_width - fg_w) // 2, (bg_height - fg_h) // 2
                
                src_quad = [
                    (fg_x, fg_y),
                    (fg_x + fg_w, fg_y),
                    (fg_x + fg_w, fg_y + fg_h),
                    (fg_x, fg_y + fg_h)
                ]
                
                # 使用PIL进行透视变换
                coeffs = self.find_coeffs(dst_quad, src_quad)
                
                # 创建一个临时透明图像
                temp = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
                
                # 调整前景大小
                fg_resized = fg_pil.resize((fg_w, fg_h), Image.LANCZOS)
                if fg_resized.mode != 'RGBA':
                    fg_resized = fg_resized.convert('RGBA')
                
                # 将调整大小后的前景图像放在临时画布上
                temp.paste(fg_resized, (fg_x, fg_y))
                
                # 应用透视变换
                transformed = temp.transform((bg_width, bg_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
                
                # 将变换后的图像合成到结果上
                result_pil = Image.alpha_composite(result_pil.convert('RGBA'), transformed)
            
            # 转换回PyTorch张量
            result_np = np.array(result_pil.convert('RGB'))
            result_tensor = torch.from_numpy(result_np).float() / 255.0
            result_tensor = result_tensor.unsqueeze(0)
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"预设变换错误: {str(e)}")
            traceback.print_exc()
            return (背景图像,)
    
    def find_coeffs(self, pa, pb):
        """计算透视变换的系数"""
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(sum([[p[0], p[1]] for p in pb], []), dtype=np.float32)
        
        res = np.linalg.solve(A, B)
        return np.array(res).reshape(8).tolist()

@PromptServer.instance.routes.post("/popup_blender/apply")
async def apply_popup_blender(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        image_data = data.get("image_data")
        corner_points = data.get("corner_points", [])  # 获取角点坐标
        
        if node_id not in node_data:
            return web.json_response({"success": False, "error": "节点数据不存在"})
        
        try:
            node_info = node_data[node_id]
            
            if image_data:
                # 处理base64图像数据
                if image_data.startswith('data:image'):
                    base64_data = image_data.split(',')[1]
                else:
                    base64_data = image_data
                
                image_bytes = base64.b64decode(base64_data)
                buffer = io.BytesIO(image_bytes)
                pil_image = Image.open(buffer)
                
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                
                np_image = np.array(pil_image)
                tensor_image = torch.from_numpy(np_image / 255.0).float().unsqueeze(0)
                node_info["result"] = tensor_image
            
            # 保存角点坐标
            node_info["corner_points"] = corner_points
            
            node_info["event"].set()
            return web.json_response({"success": True})
            
        except Exception as e:
            if node_id in node_data and "event" in node_data[node_id]:
                node_data[node_id]["event"].set()
            return web.json_response({"success": False, "error": str(e)})

    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})

@PromptServer.instance.routes.post("/popup_blender/cancel")
async def cancel_popup_blender(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        
        if node_id in node_data:
            node_info = node_data[node_id]
            node_info["event"].set()
            del node_data[node_id]
        
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})

class 补齐快速:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("补全结果",)
    FUNCTION = "complete_mask"
    CATEGORY = "Distortion Transformation Kit"

    # 定义参数预设值
    THRESHOLD_VALUES = [25, 50, 75]  # 低、中、高阈值
    LENGTH_VALUES = [25, 50, 100]    # 短、中、长线条
    GAP_VALUES = [5, 10, 20]         # 小、中、大间隙
    SMOOTH_VALUES = [True, False]     # 平滑和锐利边缘

    def detect_lines(self, mask, threshold, min_length, max_gap):
        """使用霍夫变换检测直线"""
        edges = cv2.Canny(mask, threshold//2, threshold)
        lines = cv2.HoughLinesP(edges, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=threshold,
                               minLineLength=min_length,
                               maxLineGap=max_gap)
        return lines if lines is not None else []

    def extend_line(self, x1, y1, x2, y2, shape):
        """延长线段到图像边界"""
        h, w = shape
        
        if x2 - x1 == 0:  # 垂直线
            return x1, 0, x1, h-1
            
        # 计算直线方程 y = mx + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # 延伸到x=0和x=w
        left_y = int(b)
        right_y = int(m * w + b)
        
        # 如果线超出图像范围，调整端点
        if left_y < 0:
            x_start = int(-b / m)
            y_start = 0
        elif left_y >= h:
            x_start = int((h-1 - b) / m)
            y_start = h-1
        else:
            x_start = 0
            y_start = left_y
            
        if right_y < 0:
            x_end = int(-b / m)
            y_end = 0
        elif right_y >= h:
            x_end = int((h-1 - b) / m)
            y_end = h-1
        else:
            x_end = w-1
            y_end = right_y
            
        return x_start, y_start, x_end, y_end

    def find_intersection(self, line1, line2):
        """计算两条线的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 计算分母
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:  # 平行线
            return None
            
        # 计算交点
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        
        return (int(px), int(py))

    def is_quadrilateral(self, points):
        """判断点集是否能构成四边形，并返回最佳的四边形逼近"""
        if len(points) < 4:
            return False, None
        
        # 使用凸包算法找到最外层的点
        hull = cv2.convexHull(points)
        
        # 对凸包进行多边形逼近，尝试不同的epsilon值
        best_approx = None
        best_score = float('inf')
        
        # 尝试不同的epsilon值来找到最佳的四边形逼近
        for eps_factor in [0.01, 0.02, 0.03, 0.05]:
            epsilon = eps_factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                # 计算四边形的质量分数（基于角度和边长的均匀性）
                score = self.calculate_quadrilateral_score(approx)
                if score < best_score:
                    best_score = score
                    best_approx = approx
        
        return best_approx is not None, best_approx

    def calculate_quadrilateral_score(self, quad):
        """计算四边形的质量分数（越低越好）"""
        # 计算四个角的角度
        angles = []
        for i in range(4):
            p1 = quad[i][0]
            p2 = quad[(i+1)%4][0]
            p3 = quad[(i+2)%4][0]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            angle = np.abs(np.degrees(
                np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            ))
            angles.append(angle)
        
        # 计算边长
        sides = []
        for i in range(4):
            p1 = quad[i][0]
            p2 = quad[(i+1)%4][0]
            side = np.sqrt(np.sum((p2 - p1) ** 2))
            sides.append(side)
        
        # 计算角度偏差（与90度的差异）和边长偏差
        angle_score = np.std([abs(a - 90) for a in angles])
        side_score = np.std(sides) / np.mean(sides)
        
        return angle_score + side_score * 100

    def force_quadrilateral(self, mask_binary):
        """强制将mask转换为四边形"""
        # 找到轮廓
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_binary
            
        # 获取最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # 创建新的mask
        h, w = mask_binary.shape
        new_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(new_mask, [box], 255)
        
        return new_mask

    def calculate_shape_score(self, mask, original_area):
        """计算形状得分"""
        # 计算面积比例
        current_area = np.sum(mask) / 255
        area_ratio = min(current_area / original_area, original_area / current_area)
        
        # 计算轮廓规则度
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        # 使用等周率判断形状规则度
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 综合得分
        return 0.6 * area_ratio + 0.4 * circularity

    def process_with_params(self, mask_binary, threshold, min_length, max_gap, do_smooth, h, w):
        """使用指定参数处理mask"""
        # 检测直线
        lines = self.detect_lines(mask_binary, threshold, min_length, max_gap)
        
        # 创建输出mask
        completed_mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(lines) > 0:
            # 延长所有线段
            extended_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                ext_line = self.extend_line(x1, y1, x2, y2, (h, w))
                extended_lines.append(ext_line)
            
            # 找到所有交点
            intersections = []
            for i in range(len(extended_lines)):
                for j in range(i+1, len(extended_lines)):
                    point = self.find_intersection(extended_lines[i], extended_lines[j])
                    if point is not None:
                        x, y = point
                        if 0 <= x < w and 0 <= y < h:
                            intersections.append(point)
            
            if len(intersections) >= 4:
                # 检查是否可以构成四边形
                points = np.array(intersections)
                is_quad, approx = self.is_quadrilateral(points.reshape(-1, 1, 2))
                
                if is_quad:
                    # 使用近似多边形的点绘制四边形
                    cv2.fillPoly(completed_mask, [approx], 255)
                else:
                    # 如果无法形成合适的四边形，强制转换为四边形
                    completed_mask = self.force_quadrilateral(mask_binary)
            else:
                # 如果交点不足，强制转换为四边形
                completed_mask = self.force_quadrilateral(mask_binary)
        else:
            # 如果没有检测到线段，强制转换为四边形
            completed_mask = self.force_quadrilateral(mask_binary)
            
        # 应用平滑处理
        if do_smooth:
            kernel = np.ones((3, 3), np.uint8)
            completed_mask = cv2.morphologyEx(completed_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            completed_mask = cv2.GaussianBlur(completed_mask, (3, 3), 0)
            _, completed_mask = cv2.threshold(completed_mask, 127, 255, cv2.THRESH_BINARY)
            
        return completed_mask

    def complete_mask(self, mask):
        try:
            # 将mask转换为numpy数组
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
                if len(mask_np.shape) == 3:
                    mask_np = mask_np[0]
            else:
                mask_np = np.array(mask)
            
            # 确保mask为二值图像
            mask_binary = np.where(mask_np > 0.5, 255, 0).astype(np.uint8)
            h, w = mask_binary.shape
            
            # 计算原始mask的面积
            original_area = np.sum(mask_binary) / 255
            
            # 尝试所有参数组合，找到最佳结果
            best_score = 0
            best_result = mask_binary.copy()
            
            for threshold, length, gap, smooth in itertools.product(
                self.THRESHOLD_VALUES,
                self.LENGTH_VALUES,
                self.GAP_VALUES,
                self.SMOOTH_VALUES
            ):
                result = self.process_with_params(mask_binary, threshold, length, gap, smooth, h, w)
                score = self.calculate_shape_score(result, original_area)
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_result = result
            
            # 转换结果为torch tensor
            return (torch.from_numpy(best_result.astype(np.float32) / 255.0).unsqueeze(0),)
            
        except Exception as e:
            print(f"Mask completion error: {str(e)}")
            return (mask,)

def appx_best_fit_ngon(mask_cv2, n: int = 4) -> list:
    """
    近似拟合掩码为n边形
    
    Args:
        mask_cv2: OpenCV格式的二值掩码
        n: 多边形的边数，默认为4
        
    Returns:
        n个顶点的列表，每个顶点是(x,y)坐标
    """
    # 确保掩码是二值图像
    if len(mask_cv2.shape) > 2:
        mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    else:
        mask_cv2_gray = mask_cv2
    
    # 确保像素值是0或255
    _, mask_cv2_gray = cv2.threshold(mask_cv2_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 找到轮廓
    contours, _ = cv2.findContours(
        mask_cv2_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 如果没有找到轮廓，返回空列表
    if not contours:
        return []
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算凸包
    hull = cv2.convexHull(largest_contour)
    hull = np.array(hull).reshape((len(hull), 2))
    
    # 如果凸包点数小于等于目标n，直接返回
    if len(hull) <= n:
        return hull.tolist()

    # 转换为sympy点
    hull_points = [sympy.Point(*pt) for pt in hull]
    
    # 迭代直到顶点数达到n
    while len(hull_points) > n:
        best_candidate = None

        # 对于hull中的每条边
        for edge_idx_1 in range(len(hull_points)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull_points)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull_points)
            adj_idx_2 = (edge_idx_2 + 1) % len(hull_points)

            edge_pt_1 = hull_points[edge_idx_1]
            edge_pt_2 = hull_points[edge_idx_2]
            adj_pt_1 = hull_points[adj_idx_1]
            adj_pt_2 = hull_points[adj_idx_2]

            # 创建包含相邻点的子多边形
            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # 确保内角和大于180度
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # 计算删除边后的新顶点
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)
            
            # 确保有交点
            if not intersect:
                continue
                
            intersect = intersect[0]

            # 计算删除边后增加的面积
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            
            # 选择增加面积最小的边
            if best_candidate is None or best_candidate[1] > area:
                # 创建新的hull，删除选中的边
                better_hull = list(hull_points)
                better_hull[edge_idx_1] = intersect
                del better_hull[edge_idx_2 if edge_idx_2 > edge_idx_1 else edge_idx_2 % len(better_hull)]
                best_candidate = (better_hull, area)

        if not best_candidate:
            break  # 如果找不到合适的边，中断循环

        hull_points = best_candidate[0]

    # 转换回整数坐标
    hull = [(int(float(p.x)), int(float(p.y))) for p in hull_points]
    
    return hull

def create_polygon_mask(vertices, width, height):
    """
    根据顶点创建多边形掩码
    
    Args:
        vertices: 顶点列表，每个顶点是(x,y)坐标
        width: 图像宽度
        height: 图像高度
        
    Returns:
        掩码图像，0-1范围内的浮点数，形状为(height, width)
    """
    # 创建空白图像
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 绘制多边形
    if len(vertices) > 2:  # 至少需要3个点才能形成多边形
        draw.polygon(vertices, fill=255)
    
    # 转换为numpy数组并归一化
    mask_np = np.array(mask).astype(np.float32) / 255.0
    
    return mask_np

def create_comparison_image(original_mask, vertices, width, height):
    """
    创建对比图，显示原始掩码和用红色框包裹的多边形边界
    
    Args:
        original_mask: 原始掩码，0-1范围内的浮点数
        vertices: 多边形顶点列表
        width: 图像宽度
        height: 图像高度
        
    Returns:
        对比图像，RGB格式，0-1范围内的浮点数
    """
    # 确保掩码的维度正确
    if len(original_mask.shape) == 3 and original_mask.shape[2] > 1:
        original_mask = original_mask[:, :, 0]
    
    # 创建RGB图像，将原始掩码放在所有通道上
    comparison = np.stack([original_mask] * 3, axis=2)
    
    # 创建一个空白图像用于绘制多边形边界
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 如果有足够的顶点，绘制多边形边界
    if len(vertices) > 2:
        # 绘制多边形，只绘制边界，不填充
        draw.polygon(vertices, outline=(255, 0, 0, 255), width=2, fill=None)
    
    # 将边界叠加到原始图像上
    overlay_np = np.array(overlay).astype(np.float32) / 255.0
    
    # 将红色边界叠加到图像上
    mask = overlay_np[:, :, 3] > 0
    comparison[mask, 0] = overlay_np[mask, 0]  # 红色通道
    comparison[mask, 1] = overlay_np[mask, 1]  # 绿色通道
    comparison[mask, 2] = overlay_np[mask, 2]  # 蓝色通道
    
    return comparison

class 补齐慢速自适应强:
    """
    将掩码近似为N边形的ComfyUI节点
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "sides": ("INT", {
                    "default": 4,
                    "min": 3,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "threshold": ("INT", {
                    "default": 127,
                    "min": 1,
                    "max": 254,
                    "step": 1,
                    "display": "slider",
                    "label": "二值化阈值"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("原始图像", "二值化掩码", "近似多边形掩码", "对比结果")
    FUNCTION = "process"
    CATEGORY = "Distortion Transformation Kit"

    def process(self, mask, sides, threshold):
        # 确保掩码是2D的
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask[0]
        
        # 转换为CV2格式（0-255范围）
        mask_cv2 = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        # 获取图像尺寸
        height, width = mask_cv2.shape
        
        # 二值化处理，使用用户提供的阈值
        _, binary_mask = cv2.threshold(mask_cv2, threshold, 255, cv2.THRESH_BINARY)
        binary_mask_np = binary_mask.astype(np.float32) / 255.0
        
        # 计算近似多边形（使用二值化后的掩码）
        vertices = appx_best_fit_ngon(binary_mask, sides)
        
        # 创建多边形掩码
        poly_mask_np = create_polygon_mask(vertices, width, height)
        
        # 创建对比图 - 修改为传递顶点
        comparison_np = create_comparison_image(mask.cpu().numpy(), vertices, width, height)
        
        # 准备输出图像
        # 1. 原始图像 - 转为RGB格式
        original_rgb = np.stack([mask.cpu().numpy()] * 3, axis=2)
        
        # 2. 二值化掩码图像 - 转为RGB格式
        binary_rgb = np.stack([binary_mask_np] * 3, axis=2)
        
        # 转换为ComfyUI格式
        original_tensor = torch.from_numpy(original_rgb).unsqueeze(0)
        binary_tensor = torch.from_numpy(binary_rgb).unsqueeze(0)
        poly_mask_tensor = torch.from_numpy(poly_mask_np).unsqueeze(0)  # 作为MASK输出
        comparison_tensor = torch.from_numpy(comparison_np).unsqueeze(0)
        
        return (original_tensor, binary_tensor, poly_mask_tensor, comparison_tensor)

class QuadMaskWarper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "mask": ("MASK",),
                "rotation": (["0", "90", "180", "270"], {"default": "0"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "warp_image"

    CATEGORY = "Distortion Transformation Kit"

    def _get_source_points(self, image):
        h, w = image.shape[:2]
        return np.array([
            [0, 0],     # 左上
            [w, 0],     # 右上
            [w, h],     # 右下 
            [0, h]      # 左下
        ], dtype=np.float32)

    def _find_quad_points(self, mask_np):
        # 确保遮罩是二值图像
        _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 选择最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 使用Douglas-Peucker算法简化轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果点数不是4，尝试调整epsilon
        if len(approx) != 4:
            if len(approx) > 4:
                while len(approx) > 4:
                    epsilon *= 1.1
                    approx = cv2.approxPolyDP(contour, epsilon, True)
            else:
                while len(approx) < 4:
                    epsilon *= 0.9
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if epsilon < 1e-6:
                        break
        
        if len(approx) != 4:
            # 如果仍然不是4个点，使用最小面积矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = np.array(box, dtype=np.int32)
        
        # 重新排序点，确保顺序是：左上、右上、右下、左下
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 计算质心
        center = np.mean(pts, axis=0)
        
        # 计算每个点相对于质心的角度
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        
        # 按角度排序
        sorted_idx = np.argsort(angles)
        sorted_pts = pts[sorted_idx]
        
        # 旋转点数组使得第一个点是左上角（具有最小x+y值的点）
        s = sorted_pts.sum(axis=1)
        min_idx = np.argmin(s)
        sorted_pts = np.roll(sorted_pts, -min_idx, axis=0)
        
        return sorted_pts.astype(np.float32)
    
    def warp_image(self, base_image, target_image, mask, rotation="0"):
        # 转换输入图像
        base_np = cv2.cvtColor(base_image[0].cpu().numpy() * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)
        target_np = cv2.cvtColor(target_image[0].cpu().numpy() * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)
        
        # 处理旋转
        if rotation != "0":
            angle = int(rotation)
            if angle == 90:
                target_np = cv2.rotate(target_np, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                target_np = cv2.rotate(target_np, cv2.ROTATE_180)
            elif angle == 270:
                target_np = cv2.rotate(target_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 处理遮罩
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 如果遮罩是白色区域为主，则反转
        if np.mean(mask_np) > 127:
            mask_np = 255 - mask_np
            
        # 找到四边形的点
        quad_points = self._find_quad_points(mask_np)
        
        if quad_points is None:
            print("警告: 未能找到有效的四边形区域")
            return (base_image,)
        
        # 计算透视变换矩阵
        source_points = self._get_source_points(target_np)
        matrix = cv2.getPerspectiveTransform(source_points, quad_points)
        
        # 执行透视变换
        warped = cv2.warpPerspective(
            target_np,
            matrix,
            (base_np.shape[1], base_np.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # 创建二值遮罩
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # 对遮罩进行轻微的模糊，仅用于边缘过渡
        kernel_size = 3  # 使用较小的核大小
        mask_edges = cv2.GaussianBlur(mask_binary, (kernel_size, kernel_size), 0)
        
        # 创建遮罩的边缘区域
        edge_mask = cv2.subtract(
            cv2.dilate(mask_binary, np.ones((3,3), np.uint8)),
            cv2.erode(mask_binary, np.ones((3,3), np.uint8))
        )
        
        # 转换为3通道
        mask_3d = mask_binary[:, :, np.newaxis] / 255.0
        edge_mask_3d = edge_mask[:, :, np.newaxis] / 255.0
        mask_edges_3d = mask_edges[:, :, np.newaxis] / 255.0
        
        # 在边缘区域进行平滑混合，其他区域保持原样
        result = base_np.copy()
        # 非边缘区域直接使用warped图像
        result = np.where(mask_3d > 0.5, warped, result)
        # 仅在边缘区域进行平滑过渡
        edge_blend = warped * mask_edges_3d + result * (1 - mask_edges_3d)
        result = np.where(edge_mask_3d > 0, edge_blend, result)
        
        # 转换回RGB并返回
        result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(result_rgb / 255.0).unsqueeze(0),)

WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {
    "WjbDistortionTransform": WjbDistortionTransform,
    "WjbDistortionTransformDirect": WjbDistortionTransformDirect,
    "补齐快速": 补齐快速,
    "补齐慢速自适应强": 补齐慢速自适应强,
    "QuadMaskWarper": QuadMaskWarper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WjbDistortionTransform": "扭曲变换可视化",
    "WjbDistortionTransformDirect": "预设变换",
    "补齐快速": "补齐快速",
    "补齐慢速自适应强": "补齐慢速自适应强",
    "QuadMaskWarper": "变换节点"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 