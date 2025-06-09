import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 创建弹窗式图像混合控制器
function createPopupBlenderDialog(nodeId, backgroundImage, foregroundImage) {
    // 创建模态窗口
    const modal = document.createElement("dialog");
    modal.id = "popup-blender-modal";
    modal.style.cssText = `
        padding: 0;
        background: #2a2a2a;
        border: none;
        border-radius: 8px;
        max-width: 98vw;
        max-height: 98vh;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0,0,0,0.7);
    `;
    
    // 创建内容容器
    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        min-width: 800px;
        min-height: 600px;
    `;
    
    // 创建标题栏
    const header = document.createElement("div");
    header.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 20px;
        background: #333;
        border-bottom: 1px solid #444;
    `;
    
    const title = document.createElement("h3");
    title.textContent = "扭曲变换调整";
    title.style.cssText = `
        margin: 0;
        color: #fff;
        font-size: 20px;
        font-weight: 500;
    `;
    
    const closeBtn = document.createElement("button");
    closeBtn.innerHTML = "×";
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: #fff;
        font-size: 24px;
        cursor: pointer;
    `;
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    
    // 创建内容区域
    const content = document.createElement("div");
    content.style.cssText = `
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        padding: 20px;
        gap: 16px;
        overflow: auto;
        height: calc(100% - 50px);
    `;
    
    // 添加键盘操作说明
    const keyboardHelp = document.createElement("div");
    keyboardHelp.style.cssText = `
        background: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 14px;
        margin-bottom: 8px;
        text-align: center;
    `;
    keyboardHelp.innerHTML = "提示: 点击<b>角点按钮</b>进入微调模式(所有控件会消失)，使用<b>方向键</b>精确调整位置 (按住<b>Shift</b>加速移动，再次点击按钮退出微调模式)";
    
    
    // 创建编辑画布区域
    const canvasWrapper = document.createElement("div");
    canvasWrapper.style.cssText = `
        position: relative;
        background: #1a1a1a;
        border-radius: 4px;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-grow: 1;
        min-height: 500px;
    `;
    
    // 主画布
    const canvas = document.createElement("canvas");
    canvas.id = "blend-canvas";
    canvas.style.cssText = `
        max-width: 100%;
        max-height: 80vh;
        object-fit: contain;
    `;
    
    // 创建控制点SVG覆盖层
    const controlsSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    controlsSvg.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
    `;
    controlsSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    
    // 添加到画布容器
    canvasWrapper.appendChild(canvas);
    canvasWrapper.appendChild(controlsSvg);
    
    // 创建工具栏
    const toolbar = document.createElement("div");
    toolbar.style.cssText = `
        display: flex;
        gap: 20px;
        align-items: center;
        padding: 15px 0;
        margin-top: 10px;
    `;
    
    // 角点选择按钮组
    const cornerButtonGroup = document.createElement("div");
    cornerButtonGroup.style.cssText = `
        display: flex;
        gap: 10px;
        align-items: center;
    `;
    
    const cornerLabel = document.createElement("span");
    cornerLabel.textContent = "选择角点:";
    cornerLabel.style.cssText = `
        color: #ccc;
        font-size: 14px;
    `;
    cornerButtonGroup.appendChild(cornerLabel);
    
    // 创建四个角点选择按钮
    const cornerNames = ["左上", "右上", "右下", "左下"];
    const cornerButtons = [];
    
            for (let i = 0; i < 4; i++) {
        const btn = document.createElement("button");
        btn.textContent = cornerNames[i] + " (" + (i + 1) + ")";
        btn.dataset.cornerIndex = i;
        btn.style.cssText = `
            padding: 5px 10px;
            background: #555;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            min-width: 70px;
        `;
        
        btn.addEventListener("click", () => {
            // 设置活动点
            controls.activePoint = parseInt(btn.dataset.cornerIndex);
            
            // 切换微调模式
            controls.adjustmentMode = !controls.adjustmentMode;
            
            // 更新按钮样式
            cornerButtons.forEach((button, idx) => {
                if (idx === controls.activePoint && controls.adjustmentMode) {
                    button.style.background = "#4CAF50";
                    button.style.fontWeight = "bold";
                } else {
                    button.style.background = "#555";
                    button.style.fontWeight = "normal";
                }
            });
            
            // 重绘以突出显示选中的角点
            drawForegroundAndControls();
        });
        
        cornerButtons.push(btn);
        cornerButtonGroup.appendChild(btn);
    }
    
    // 操作按钮组
    const buttonGroup = document.createElement("div");
    buttonGroup.style.cssText = `
        display: flex;
        gap: 15px;
        margin-left: auto;
    `;
    
    const resetButton = document.createElement("button");
    resetButton.textContent = "重置";
    resetButton.style.cssText = `
        padding: 8px 16px;
        background: #555;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    `;
    
    const confirmButton = document.createElement("button");
    confirmButton.textContent = "确认";
    confirmButton.style.cssText = `
        padding: 10px 20px;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    `;
    
    const cancelButton = document.createElement("button");
    cancelButton.textContent = "取消";
    cancelButton.style.cssText = `
        padding: 10px 20px;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    `;
    
    buttonGroup.appendChild(resetButton);
    buttonGroup.appendChild(confirmButton);
    buttonGroup.appendChild(cancelButton);
    
    // 将工具栏组件添加到工具栏
    toolbar.appendChild(cornerButtonGroup);
    toolbar.appendChild(buttonGroup);
    
    // 组装界面
    content.appendChild(keyboardHelp);
    content.appendChild(canvasWrapper);
    content.appendChild(toolbar);
    
    container.appendChild(header);
    container.appendChild(content);
    
    modal.appendChild(container);
    document.body.appendChild(modal);
    
    // 控制状态
    const controls = {
        points: [],       // 四个角点的位置
        originalPoints: [], // 原始四个角点位置
        bgImage: null,    // 背景图像
        fgImage: null,    // 前景图像
        handleSize: 12,    // 控制点大小
        activePoint: -1,  // 当前活动点索引（由按钮选择）
        dragPointIndex: -1, // 当前正在拖动的点索引
        isDragging: false, // 是否正在拖动
        isDraggingImage: false, // 是否在拖动整个图像
        adjustmentMode: false, // 是否处于微调模式（控件隐藏）
        lastMouseX: 0,    // 上次鼠标X位置
        lastMouseY: 0,    // 上次鼠标Y位置
        transform: {      // 变换参数
            matrix: null  // 变换矩阵
        },
        originalWidth: 0,  // 原始背景图像宽度
        originalHeight: 0  // 原始背景图像高度
    };
    
    // 初始化图像和控制点
    function initImageAndControls() {
        // 加载背景图像
        const bgImg = new Image();
        bgImg.onload = () => {
            controls.bgImage = bgImg;
            controls.originalWidth = bgImg.width;  // 保存原始宽度
            controls.originalHeight = bgImg.height;  // 保存原始高度
            
            // 如果前景图像已加载，则初始化画布
            if (controls.fgImage) {
                initCanvas();
            }
        };
        bgImg.src = backgroundImage;
        
        // 加载前景图像
        const fgImg = new Image();
        fgImg.onload = () => {
            controls.fgImage = fgImg;
            
            // 如果背景图像已加载，则初始化画布
            if (controls.bgImage) {
                initCanvas();
            }
        };
        fgImg.src = foregroundImage;
    }
    
    // 初始化画布
    function initCanvas() {
        // 设置画布尺寸
        const bgAspectRatio = controls.bgImage.width / controls.bgImage.height;
        
        // 计算适合窗口的尺寸
        const maxWidth = Math.min(window.innerWidth * 0.8, 1200);
        const maxHeight = Math.min(window.innerHeight * 0.7, 800);
        
        let width, height;
        
        if (bgAspectRatio > 1) {
            // 宽大于高
            width = Math.min(maxWidth, controls.bgImage.width);
            height = width / bgAspectRatio;
            
            if (height > maxHeight) {
                height = maxHeight;
                width = height * bgAspectRatio;
            }
        } else {
            // 高大于宽
            height = Math.min(maxHeight, controls.bgImage.height);
            width = height * bgAspectRatio;
            
            if (width > maxWidth) {
                width = maxWidth;
                height = width / bgAspectRatio;
            }
        }
        
        // 设置画布尺寸
        canvas.width = width;
        canvas.height = height;
        
        // 设置SVG尺寸
        controlsSvg.setAttribute("width", width);
        controlsSvg.setAttribute("height", height);
        controlsSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        
        // 绘制背景图像
        const ctx = canvas.getContext("2d");
        ctx.drawImage(controls.bgImage, 0, 0, width, height);
        
        // 计算前景图像的初始位置（居中）
        const fgScaleFactor = Math.min(width * 0.8 / controls.fgImage.width, height * 0.8 / controls.fgImage.height);
        const fgWidth = controls.fgImage.width * fgScaleFactor;
        const fgHeight = controls.fgImage.height * fgScaleFactor;
        const fgX = (width - fgWidth) / 2;
        const fgY = (height - fgHeight) / 2;
        
        // 初始化控制点位置
        controls.points = [
            { x: fgX, y: fgY },                    // 左上
            { x: fgX + fgWidth, y: fgY },         // 右上
            { x: fgX + fgWidth, y: fgY + fgHeight }, // 右下
            { x: fgX, y: fgY + fgHeight }          // 左下
        ];
        
        // 保存原始控制点位置
        controls.originalPoints = JSON.parse(JSON.stringify(controls.points));
        
        // 绘制前景图像与控制点
        drawForegroundAndControls();
        
        // 显示模态窗口
        modal.showModal();
    }
    
    // 绘制前景图像与控制点
    function drawForegroundAndControls() {
        const ctx = canvas.getContext("2d");
        
        // 重绘背景
        ctx.drawImage(controls.bgImage, 0, 0, canvas.width, canvas.height);
        
        // 应用透视变换并绘制前景图像
        applyPerspectiveTransform();
        
        // 清空SVG
        while (controlsSvg.firstChild) {
            controlsSvg.removeChild(controlsSvg.firstChild);
        }
        
        // 如果处于微调模式，不显示任何控制点
        if (controls.adjustmentMode && controls.activePoint >= 0) {
            // 完全不显示任何标记，保持画面清晰
            return; // 不绘制任何控制点
        }
        
        // 绘制连接线 - 四边形
        const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        const pointsAttr = controls.points.map(p => `${p.x},${p.y}`).join(" ");
        polygon.setAttribute("points", pointsAttr);
        polygon.setAttribute("fill", "rgba(0, 200, 255, 0.1)");
        polygon.setAttribute("stroke", "rgba(0, 200, 255, 0.8)");
        polygon.setAttribute("stroke-width", "1.5");
        polygon.setAttribute("stroke-dasharray", "5,3");
        controlsSvg.appendChild(polygon);
        
        // 绘制控制点 - 四个角点
        controls.points.forEach((point, index) => {
            // 创建控制点
            const handle = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            const size = controls.handleSize;
            
            handle.setAttribute("x", point.x - size/2);
            handle.setAttribute("y", point.y - size/2);
            handle.setAttribute("width", size);
            handle.setAttribute("height", size);
            
            // 所有角点使用统一样式
            handle.setAttribute("fill", "rgba(0, 200, 255, 0.8)");
            handle.setAttribute("stroke", "white");
            handle.setAttribute("stroke-width", "1.5");
            
            handle.setAttribute("class", "control-handle");
            handle.setAttribute("data-index", index);
            handle.style.cursor = "move";
            handle.style.pointerEvents = "auto";
            
            // 添加拖动事件 - 允许拖动但不改变活动点状态
            handle.addEventListener("mousedown", (e) => {
                controls.isDragging = true;
                controls.dragPointIndex = index; // 记录当前拖动的是哪个点
                // 不改变activePoint状态，保持当前按钮选择的活动点
                e.preventDefault();
                e.stopPropagation();
            });
            
            controlsSvg.appendChild(handle);
            
            // 添加小标签以便识别角点
            const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
            label.setAttribute("x", point.x);
            label.setAttribute("y", point.y - size/2 - 5);
            label.setAttribute("text-anchor", "middle");
            label.setAttribute("fill", "white");
            label.setAttribute("stroke", "black");
            label.setAttribute("stroke-width", "0.5");
            label.setAttribute("font-size", "12px");
            label.textContent = (index + 1).toString();
            controlsSvg.appendChild(label);
        });
    }
    
    // 添加一个图像边缘平滑函数
    function smoothEdges(imageData, width, height) {
        // 创建临时数组存储结果
        const tempData = new Uint8ClampedArray(imageData.data.length);
        for (let i = 0; i < imageData.data.length; i++) {
            tempData[i] = imageData.data[i];
        }
        
        // 高斯模糊卷积核
        const kernel = [
            [0.0625, 0.125, 0.0625],
            [0.125,  0.25,  0.125],
            [0.0625, 0.125, 0.0625]
        ];
        
        // 仅处理边缘区域（部分透明的像素）
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                const alpha = imageData.data[idx + 3];
                
                // 只处理半透明区域（边缘）
                if (alpha > 5 && alpha < 250) {
                    // 对边缘像素应用高斯模糊
                    for (let c = 0; c < 4; c++) { // 处理所有通道，包括alpha
                        let sum = 0;
                        let weightSum = 0;
                        
                        for (let ky = -1; ky <= 1; ky++) {
                            for (let kx = -1; kx <= 1; kx++) {
                                const sampleX = Math.min(Math.max(x + kx, 0), width - 1);
                                const sampleY = Math.min(Math.max(y + ky, 0), height - 1);
                                const sampleIdx = (sampleY * width + sampleX) * 4 + c;
                                const weight = kernel[ky + 1][kx + 1];
                                
                                sum += imageData.data[sampleIdx] * weight;
                                weightSum += weight;
                            }
                        }
                        
                        tempData[idx + c] = sum / weightSum;
                    }
                }
            }
        }
        
        // 将平滑结果复制回原始数据
        for (let i = 0; i < imageData.data.length; i++) {
            imageData.data[i] = tempData[i];
        }
        
        return imageData;
    }
    
    // 修改applyPerspectiveTransform函数中的代码，在画布绘制前平滑边缘
    function applyPerspectiveTransform() {
        const ctx = canvas.getContext("2d");
        
        // 清除画布并重新绘制背景图像
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(controls.bgImage, 0, 0, canvas.width, canvas.height);
        
        // 检查控制点是否被移动
        let hasChanges = false;
        for (let i = 0; i < controls.points.length; i++) {
            const p = controls.points[i];
            const op = controls.originalPoints[i];
            if (p.x !== op.x || p.y !== op.y) {
                hasChanges = true;
                break;
            }
        }
        
        // 计算前景图像的初始尺寸
        const fgScaleFactor = Math.min(canvas.width * 0.8 / controls.fgImage.width, canvas.height * 0.8 / controls.fgImage.height);
        const fgWidth = controls.fgImage.width * fgScaleFactor;
        const fgHeight = controls.fgImage.height * fgScaleFactor;
        
        // 如果没有变化，直接绘制前景图像在原始位置
        if (!hasChanges) {
            const fgX = (canvas.width - fgWidth) / 2;
            const fgY = (canvas.height - fgHeight) / 2;
            ctx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
            return;
        }
        
        // 检查四边形是否有效（凸四边形，没有交叉边）
        if (!isValidQuadrilateral(controls.points)) {
            // 如果四边形无效，回退到使用原始矩形
            console.warn("无效的四边形变换，使用正常矩形");
            const fgX = (canvas.width - fgWidth) / 2;
            const fgY = (canvas.height - fgHeight) / 2;
            ctx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
            return;
        }
        
        // 使用Web API的高级变换方法
        try {
            // 尝试使用Canvas API的变换功能
            ctx.save();
            
            // 临时画布用于隔离前景图像处理
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = canvas.width;
            tempCanvas.height = canvas.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // 绘制前景图到临时画布上的原始位置
            const fgX = (canvas.width - fgWidth) / 2;
            const fgY = (canvas.height - fgHeight) / 2;
            tempCtx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
            
            // 获取前景图像数据
            const imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
            
            // 创建新的图像数据，但不覆盖整个画布
            const resultData = ctx.createImageData(canvas.width, canvas.height);
            
            // 先获取当前画布（已经绘制了背景）的像素数据
            const currentData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            // 复制当前画布数据到结果数据（保留背景）
            for (let i = 0; i < currentData.data.length; i++) {
                resultData.data[i] = currentData.data[i];
            }
            
            // 源四边形（前景图像原始矩形）
            const srcQuad = [
                { x: fgX, y: fgY },
                { x: fgX + fgWidth, y: fgY },
                { x: fgX + fgWidth, y: fgY + fgHeight },
                { x: fgX, y: fgY + fgHeight }
            ];
            
            // 目标四边形（变换后的位置）
            const dstQuad = controls.points;
            
            // 计算投影变换矩阵 (从目标到源的反向映射)
            const M = computeProjectiveTransform(dstQuad, srcQuad);
            
            if (!M) {
                console.error("计算投影变换矩阵失败，回退到简单绘制");
                ctx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
                return;
            }
            
            // 使用矩阵进行精确的像素级投影变换
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    // 检查点是否在目标四边形内部
                    if (isPointInPolygon(x, y, dstQuad)) {
                        // 计算源坐标
                        const srcPoint = applyTransform(M, x, y);
                        
                        // 检查源坐标是否在前景图像范围内
                        if (
                            srcPoint.x >= 0 && srcPoint.x < canvas.width &&
                            srcPoint.y >= 0 && srcPoint.y < canvas.height
                        ) {
                            // 双线性插值
                            const x0 = Math.floor(srcPoint.x);
                            const y0 = Math.floor(srcPoint.y);
                            const x1 = Math.min(Math.ceil(srcPoint.x), canvas.width - 1);
                            const y1 = Math.min(Math.ceil(srcPoint.y), canvas.height - 1);
                            
                            // 计算权重
                            const wx = srcPoint.x - x0;
                            const wy = srcPoint.y - y0;
                            
                            // 计算四个像素点的索引
                            const idx = (y * canvas.width + x) * 4;
                            const idx00 = (y0 * canvas.width + x0) * 4;
                            const idx01 = (y0 * canvas.width + x1) * 4;
                            const idx10 = (y1 * canvas.width + x0) * 4;
                            const idx11 = (y1 * canvas.width + x1) * 4;
                            
                            // 双线性插值计算颜色 - 检查每个索引是否有效
                            for (let c = 0; c < 3; c++) {  // 只处理RGB通道
                                const color00 = (idx00 >= 0 && idx00 < imageData.data.length) ? imageData.data[idx00 + c] : 0;
                                const color01 = (idx01 >= 0 && idx01 < imageData.data.length) ? imageData.data[idx01 + c] : 0;
                                const color10 = (idx10 >= 0 && idx10 < imageData.data.length) ? imageData.data[idx10 + c] : 0;
                                const color11 = (idx11 >= 0 && idx11 < imageData.data.length) ? imageData.data[idx11 + c] : 0;
                                
                                // 双线性插值计算
                                const colorTop = color00 * (1 - wx) + color01 * wx;
                                const colorBottom = color10 * (1 - wx) + color11 * wx;
                                const color = colorTop * (1 - wy) + colorBottom * wy;
                                
                                resultData.data[idx + c] = color;
                            }
                            
                            // 单独处理Alpha通道
                            let alpha00 = (idx00 >= 0 && idx00 < imageData.data.length) ? imageData.data[idx00 + 3] : 0;
                            let alpha01 = (idx01 >= 0 && idx01 < imageData.data.length) ? imageData.data[idx01 + 3] : 0;
                            let alpha10 = (idx10 >= 0 && idx10 < imageData.data.length) ? imageData.data[idx10 + 3] : 0;
                            let alpha11 = (idx11 >= 0 && idx11 < imageData.data.length) ? imageData.data[idx11 + 3] : 0;
                            
                            // 检查是否在原始前景矩形内
                            const isInOriginalFg = 
                                srcPoint.x >= fgX && srcPoint.x <= fgX + fgWidth &&
                                srcPoint.y >= fgY && srcPoint.y <= fgY + fgHeight;
                                
                            // 如果在原始前景矩形内，增强Alpha值
                            if (isInOriginalFg) {
                                // 增强原始区域边缘的Alpha，防止边缘处理不当
                                const edgeDistance = Math.min(
                                    srcPoint.x - fgX,
                                    fgX + fgWidth - srcPoint.x,
                                    srcPoint.y - fgY,
                                    fgY + fgHeight - srcPoint.y
                                );
                                
                                // 边缘增强
                                if (edgeDistance < 5) {
                                    const boost = 0.5 + edgeDistance / 10;
                                    alpha00 = Math.min(255, alpha00 * boost);
                                    alpha01 = Math.min(255, alpha01 * boost);
                                    alpha10 = Math.min(255, alpha10 * boost);
                                    alpha11 = Math.min(255, alpha11 * boost);
                                }
                            }
                            
                            // 计算插值后的Alpha值
                            const alphaTop = alpha00 * (1 - wx) + alpha01 * wx;
                            const alphaBottom = alpha10 * (1 - wx) + alpha11 * wx;
                            const alpha = alphaTop * (1 - wy) + alphaBottom * wy;
                            
                            // 确保前景图像可见度 - 防止变黑
                            if (alpha > 0) {
                                // 使用Alpha值设置前景透明度
                                resultData.data[idx + 3] = alpha;
                                
                                // 如果前景图像原来是完全透明区域，强制设置高透明度以保证可见性
                                if (alpha < 50) {
                                    for (let c = 0; c < 3; c++) {
                                        // 避免黑色区域 - 确保RGB值不会太低
                                        if (resultData.data[idx + c] < 30) {
                                            resultData.data[idx + c] = 30;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // 在绘制到画布前应用边缘平滑
            smoothEdges(resultData, canvas.width, canvas.height);
            
            // 将结果绘制到画布
            ctx.putImageData(resultData, 0, 0);
            ctx.restore();
            
        } catch (err) {
            console.error("高级变换失败，回退到基本变换", err);
            
            // 回退到基本绘制
            const fgX = (canvas.width - fgWidth) / 2;
            const fgY = (canvas.height - fgHeight) / 2;
            ctx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
        }
    }
    
    // 检查四边形是否有效（凸四边形，没有交叉边）
    function isValidQuadrilateral(points) {
        if (points.length !== 4) return false;
        
        // 检查是否有边相交
        const edges = [
            [points[0], points[1]],
            [points[1], points[2]],
            [points[2], points[3]],
            [points[3], points[0]]
        ];
        
        for (let i = 0; i < edges.length; i++) {
            for (let j = i + 2; j < edges.length; j++) {
                // 跳过相邻边检查
                if ((i === 0 && j === 3) || Math.abs(i - j) < 2) continue;
                
                if (doLinesIntersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1])) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // 检查两条线段是否相交
    function doLinesIntersect(p1, p2, p3, p4) {
        function orientation(p, q, r) {
            const val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
            if (val === 0) return 0;  // 共线
            return (val > 0) ? 1 : 2; // 顺时针或逆时针
        }
        
        function onSegment(p, q, r) {
            return (q.x <= Math.max(p.x, r.x) && q.x >= Math.min(p.x, r.x) &&
                   q.y <= Math.max(p.y, r.y) && q.y >= Math.min(p.y, r.y));
        }
        
        const o1 = orientation(p1, p2, p3);
        const o2 = orientation(p1, p2, p4);
        const o3 = orientation(p3, p4, p1);
        const o4 = orientation(p3, p4, p2);
        
        // 一般情况
        if (o1 !== o2 && o3 !== o4) return true;
        
        // 特殊情况
        if (o1 === 0 && onSegment(p1, p3, p2)) return true;
        if (o2 === 0 && onSegment(p1, p4, p2)) return true;
        if (o3 === 0 && onSegment(p3, p1, p4)) return true;
        if (o4 === 0 && onSegment(p3, p2, p4)) return true;
        
        return false;
    }
    
    // 计算投影变换矩阵
    function computeProjectiveTransform(srcQuad, dstQuad) {
        // 源四边形的四个点
        const x0 = srcQuad[0].x, y0 = srcQuad[0].y;
        const x1 = srcQuad[1].x, y1 = srcQuad[1].y;
        const x2 = srcQuad[2].x, y2 = srcQuad[2].y;
        const x3 = srcQuad[3].x, y3 = srcQuad[3].y;
        
        // 目标四边形的四个点
        const u0 = dstQuad[0].x, v0 = dstQuad[0].y;
        const u1 = dstQuad[1].x, v1 = dstQuad[1].y;
        const u2 = dstQuad[2].x, v2 = dstQuad[2].y;
        const u3 = dstQuad[3].x, v3 = dstQuad[3].y;
        
        // 设置矩阵方程的系数
        const A = [
            [x0, y0, 1, 0, 0, 0, -u0*x0, -u0*y0],
            [0, 0, 0, x0, y0, 1, -v0*x0, -v0*y0],
            [x1, y1, 1, 0, 0, 0, -u1*x1, -u1*y1],
            [0, 0, 0, x1, y1, 1, -v1*x1, -v1*y1],
            [x2, y2, 1, 0, 0, 0, -u2*x2, -u2*y2],
            [0, 0, 0, x2, y2, 1, -v2*x2, -v2*y2],
            [x3, y3, 1, 0, 0, 0, -u3*x3, -u3*y3],
            [0, 0, 0, x3, y3, 1, -v3*x3, -v3*y3]
        ];
        
        // 设置右侧向量
        const b = [u0, v0, u1, v1, u2, v2, u3, v3];
        
        // 解线性方程组
        const h = solveLinearSystem(A, b);
        
        if (!h) {
            console.error("计算投影变换矩阵失败");
            return null;
        }
        
        // 投影变换矩阵
        return [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1]
        ];
    }
    
    // 应用投影变换到点
    function applyTransform(M, x, y) {
        if (!M) return { x, y }; // 如果没有变换矩阵，返回原点
        
        const w = M[2][0] * x + M[2][1] * y + M[2][2];
        if (Math.abs(w) < 1e-10) {
            return { x: 0, y: 0 }; // 防止除以接近零的数
        }
        
        return {
            x: (M[0][0] * x + M[0][1] * y + M[0][2]) / w,
            y: (M[1][0] * x + M[1][1] * y + M[1][2]) / w
        };
    }
    
    // 使用高斯消元法求解线性方程组
    function solveLinearSystem(A, b) {
        const n = b.length;
        const augmentedMatrix = A.map((row, i) => [...row, b[i]]);
        
        // 高斯消元法求解
        for (let i = 0; i < n; i++) {
            // 寻找主元
            let maxRow = i;
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(augmentedMatrix[j][i]) > Math.abs(augmentedMatrix[maxRow][i])) {
                    maxRow = j;
                }
            }
            
            // 交换行
            if (maxRow !== i) {
                [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];
            }
            
            // 检查是否有解
            if (Math.abs(augmentedMatrix[i][i]) < 1e-10) {
                console.warn("Linear system is singular or nearly singular");
                return null;
            }
            
            // 归一化主行
            const pivot = augmentedMatrix[i][i];
            for (let j = i; j <= n; j++) {
                augmentedMatrix[i][j] /= pivot;
            }
            
            // 消元
            for (let j = 0; j < n; j++) {
                if (j !== i) {
                    const factor = augmentedMatrix[j][i];
                    for (let k = i; k <= n; k++) {
                        augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
                    }
                }
            }
        }
        
        // 提取解
        return augmentedMatrix.map(row => row[n]);
    }
    
    // 处理拖动控制点
    document.addEventListener("mousemove", (e) => {
        if (controls.isDragging && controls.dragPointIndex !== -1) {
            const rect = canvas.getBoundingClientRect();
            
            // 计算新位置
            const newX = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
            const newY = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));
            
            // 更新控制点位置
            controls.points[controls.dragPointIndex].x = newX;
            controls.points[controls.dragPointIndex].y = newY;
            
            // 重绘前景图像和控制点
            drawForegroundAndControls();
        } else if (controls.isDraggingImage) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // 计算鼠标移动距离
            const deltaX = mouseX - controls.lastMouseX;
            const deltaY = mouseY - controls.lastMouseY;
            
            // 更新所有控制点位置
            for (let i = 0; i < controls.points.length; i++) {
                controls.points[i].x += deltaX;
                controls.points[i].y += deltaY;
            }
            
            // 更新上次鼠标位置
            controls.lastMouseX = mouseX;
            controls.lastMouseY = mouseY;
            
            // 重绘前景图像和控制点
            drawForegroundAndControls();
        }
    });
    
    document.addEventListener("mouseup", () => {
        controls.isDragging = false;
        controls.isDraggingImage = false;
        controls.dragPointIndex = -1; // 清除拖动点索引，但保留活动点索引
    });
    
    // 允许拖动整个图像
    canvas.addEventListener("mousedown", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // 检查是否点击在四边形内部
        if (isPointInPolygon(mouseX, mouseY, controls.points)) {
            controls.isDraggingImage = true;
            controls.lastMouseX = mouseX;
            controls.lastMouseY = mouseY;
            e.preventDefault();
        }
    });
    
    // 检查点是否在多边形内部
    function isPointInPolygon(x, y, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            
            const intersect = ((yi > y) !== (yj > y)) && 
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }
    
    // 重置按钮
    resetButton.addEventListener("click", () => {
        // 重置所有控制点到原始位置
        controls.points = JSON.parse(JSON.stringify(controls.originalPoints));
        
        // 重绘前景图像和控制点
        drawForegroundAndControls();
    });
    
    // 确认按钮
    confirmButton.addEventListener("click", async () => {
        try {
            // 创建一个新画布，保持原始背景图像的分辨率
            const finalCanvas = document.createElement('canvas');
            finalCanvas.width = controls.originalWidth;
            finalCanvas.height = controls.originalHeight;
            const finalCtx = finalCanvas.getContext('2d');
            
            // 绘制背景图像到最终画布（使用原始尺寸）
            finalCtx.drawImage(controls.bgImage, 0, 0, controls.originalWidth, controls.originalHeight);
            
            // 计算缩放比例
            const scaleX = controls.originalWidth / canvas.width;
            const scaleY = controls.originalHeight / canvas.height;
            
            // 缩放控制点到原始图像分辨率
            const scaledPoints = controls.points.map(p => ({
                x: p.x * scaleX,
                y: p.y * scaleY
            }));
            
            // 检查控制点是否被移动
            let hasChanges = false;
            for (let i = 0; i < controls.points.length; i++) {
                const p = controls.points[i];
                const op = controls.originalPoints[i];
                if (p.x !== op.x || p.y !== op.y) {
                    hasChanges = true;
                    break;
                }
            }
            
            // 检查四边形是否有效
            const isValidTransform = isValidQuadrilateral(scaledPoints);
            
            // 如果有变化且四边形有效，应用透视变换到最终画布
            if (hasChanges && isValidTransform) {
                // 计算前景图像的初始尺寸（在原始分辨率下）
                const fgScaleFactor = Math.min(controls.originalWidth * 0.8 / controls.fgImage.width, 
                                          controls.originalHeight * 0.8 / controls.fgImage.height);
                const fgWidth = controls.fgImage.width * fgScaleFactor;
                const fgHeight = controls.fgImage.height * fgScaleFactor;
                
                // 临时画布用于隔离前景图像处理
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = controls.originalWidth;
                tempCanvas.height = controls.originalHeight;
                const tempCtx = tempCanvas.getContext('2d');
                
                // 绘制前景图到临时画布上的原始位置
                const fgX = (controls.originalWidth - fgWidth) / 2;
                const fgY = (controls.originalHeight - fgHeight) / 2;
                tempCtx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
                
                // 获取前景图像数据
                const imageData = tempCtx.getImageData(0, 0, controls.originalWidth, controls.originalHeight);
                
                // 创建新的图像数据
                const resultData = finalCtx.createImageData(controls.originalWidth, controls.originalHeight);
                
                // 先获取当前最终画布数据
                const currentData = finalCtx.getImageData(0, 0, controls.originalWidth, controls.originalHeight);
                
                // 复制当前画布数据到结果数据
                for (let i = 0; i < currentData.data.length; i++) {
                    resultData.data[i] = currentData.data[i];
                }
                
                // 源四边形（前景图像原始矩形）
                const srcQuad = [
                    { x: fgX, y: fgY },
                    { x: fgX + fgWidth, y: fgY },
                    { x: fgX + fgWidth, y: fgY + fgHeight },
                    { x: fgX, y: fgY + fgHeight }
                ];
                
                // 目标四边形（变换后的位置，缩放到原始分辨率）
                const dstQuad = scaledPoints;
                
                // 计算投影变换矩阵
                const M = computeProjectiveTransform(dstQuad, srcQuad);
                
                if (M) {
                    // 使用矩阵进行投影变换
                    for (let y = 0; y < controls.originalHeight; y++) {
                        for (let x = 0; x < controls.originalWidth; x++) {
                            // 检查点是否在目标四边形内部
                            if (isPointInPolygon(x, y, dstQuad)) {
                                // 计算源坐标
                                const srcPoint = applyTransform(M, x, y);
                                
                                // 检查源坐标是否在前景图像范围内
                                if (
                                    srcPoint.x >= 0 && srcPoint.x < controls.originalWidth &&
                                    srcPoint.y >= 0 && srcPoint.y < controls.originalHeight
                                ) {
                                    // 双线性插值
                                    const x0 = Math.floor(srcPoint.x);
                                    const y0 = Math.floor(srcPoint.y);
                                    const x1 = Math.min(Math.ceil(srcPoint.x), controls.originalWidth - 1);
                                    const y1 = Math.min(Math.ceil(srcPoint.y), controls.originalHeight - 1);
                                    
                                    // 计算权重
                                    const wx = srcPoint.x - x0;
                                    const wy = srcPoint.y - y0;
                                    
                                    // 计算四个像素点的索引
                                    const idx = (y * controls.originalWidth + x) * 4;
                                    const idx00 = (y0 * controls.originalWidth + x0) * 4;
                                    const idx01 = (y0 * controls.originalWidth + x1) * 4;
                                    const idx10 = (y1 * controls.originalWidth + x0) * 4;
                                    const idx11 = (y1 * controls.originalWidth + x1) * 4;
                                    
                                    // 检查索引是否有效
                                    const isValid00 = idx00 >= 0 && idx00 < imageData.data.length;
                                    const isValid01 = idx01 >= 0 && idx01 < imageData.data.length;
                                    const isValid10 = idx10 >= 0 && idx10 < imageData.data.length;
                                    const isValid11 = idx11 >= 0 && idx11 < imageData.data.length;
                                    
                                    // 双线性插值计算颜色
                                    for (let c = 0; c < 3; c++) {
                                        const color00 = isValid00 ? imageData.data[idx00 + c] : 0;
                                        const color01 = isValid01 ? imageData.data[idx01 + c] : 0;
                                        const color10 = isValid10 ? imageData.data[idx10 + c] : 0;
                                        const color11 = isValid11 ? imageData.data[idx11 + c] : 0;
                                        
                                        // 双线性插值计算
                                        const colorTop = color00 * (1 - wx) + color01 * wx;
                                        const colorBottom = color10 * (1 - wx) + color11 * wx;
                                        const color = colorTop * (1 - wy) + colorBottom * wy;
                                        
                                        resultData.data[idx + c] = color;
                                    }
                                    
                                    // 单独处理Alpha通道
                                    const alpha00 = isValid00 ? imageData.data[idx00 + 3] : 0;
                                    const alpha01 = isValid01 ? imageData.data[idx01 + 3] : 0;
                                    const alpha10 = isValid10 ? imageData.data[idx10 + 3] : 0;
                                    const alpha11 = isValid11 ? imageData.data[idx11 + 3] : 0;
                                    
                                    // 计算插值后的Alpha值
                                    const alphaTop = alpha00 * (1 - wx) + alpha01 * wx;
                                    const alphaBottom = alpha10 * (1 - wx) + alpha11 * wx;
                                    const alpha = alphaTop * (1 - wy) + alphaBottom * wy;
                                    
                                    // 检查是否在原始前景矩形内
                                    const isInOriginalFg = 
                                        srcPoint.x >= fgX && srcPoint.x <= fgX + fgWidth &&
                                        srcPoint.y >= fgY && srcPoint.y <= fgY + fgHeight;
                                        
                                    // 如果在原始前景矩形内，确保前景图像可见
                                    if (isInOriginalFg && alpha > 0) {
                                        resultData.data[idx + 3] = alpha;
                                        
                                        if (alpha < 50) {
                                            for (let c = 0; c < 3; c++) {
                                                if (resultData.data[idx + c] < 30) {
                                                    resultData.data[idx + c] = 30;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // 应用边缘平滑处理
                    smoothEdges(resultData, controls.originalWidth, controls.originalHeight);
                    
                    // 将结果绘制到最终画布
                    finalCtx.putImageData(resultData, 0, 0);
                } else {
                    // 如果变换矩阵计算失败，回退到基本绘制
                    console.warn("变换矩阵计算失败，回退到基本绘制");
                    tempCtx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
                    finalCtx.drawImage(tempCanvas, 0, 0);
                }
            } else if (hasChanges) {
                // 如果变换无效，回退到基本绘制
                console.warn("变换无效，回退到基本绘制");
                
                // 计算前景图像的初始尺寸
                const fgScaleFactor = Math.min(controls.originalWidth * 0.8 / controls.fgImage.width, 
                                          controls.originalHeight * 0.8 / controls.fgImage.height);
                const fgWidth = controls.fgImage.width * fgScaleFactor;
                const fgHeight = controls.fgImage.height * fgScaleFactor;
                
                // 计算居中位置
                const fgX = (controls.originalWidth - fgWidth) / 2;
                const fgY = (controls.originalHeight - fgHeight) / 2;
                
                // 直接绘制前景图像
                finalCtx.drawImage(controls.fgImage, 0, 0, controls.fgImage.width, controls.fgImage.height, fgX, fgY, fgWidth, fgHeight);
            }
            
            // 获取最终合成的图像数据
            const transformedImageData = finalCanvas.toDataURL('image/png');
            
            // 准备角点坐标数据 - 将画布坐标缩放至原始图像大小
            const cornerPoints = scaledPoints.map(point => {
                return {
                    x: Math.round(point.x),
                    y: Math.round(point.y)
                };
            });
            
            // 发送到后端
            await api.fetchApi('/popup_blender/apply', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    node_id: nodeId,
                    image_data: transformedImageData,
                    corner_points: cornerPoints
                })
            });
            
            // 关闭模态窗口
            modal.close();
            document.body.removeChild(modal);
        } catch (error) {
            console.error('错误：', error);
        }
    });
    
    // 取消按钮
    cancelButton.addEventListener("click", async () => {
        try {
            await api.fetchApi('/popup_blender/cancel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    node_id: nodeId
                })
            });
            
            // 关闭模态窗口
            modal.close();
            document.body.removeChild(modal);
        } catch (error) {
            console.error('错误：', error);
        }
    });
    
    // 关闭按钮
    closeBtn.addEventListener("click", async () => {
        try {
            await api.fetchApi('/popup_blender/cancel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    node_id: nodeId
                })
            });
            
            // 关闭模态窗口
            modal.close();
            document.body.removeChild(modal);
        } catch (error) {
            console.error('错误：', error);
        }
    });
    
    // 键盘事件处理（ESC键和方向键）
    modal.addEventListener("keydown", async (e) => {
        if (e.key === "Escape") {
            // ESC键可以退出微调模式或关闭窗口
            if (controls.adjustmentMode) {
                // 如果在微调模式，先退出微调模式
                controls.adjustmentMode = false;
                
                // 更新按钮样式
                cornerButtons.forEach(button => {
                    button.style.background = "#555";
                    button.style.fontWeight = "normal";
                });
                
                // 重绘
                drawForegroundAndControls();
                e.preventDefault();
            } else {
                // 否则关闭窗口
                e.preventDefault();
                try {
                    await api.fetchApi('/popup_blender/cancel', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            node_id: nodeId
                        })
                    });
                    
                    // 关闭模态窗口
                    modal.close();
                    document.body.removeChild(modal);
                } catch (error) {
                    console.error('错误：', error);
                }
            }
        } else if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
            e.preventDefault(); // 防止页面滚动
            
            // 只有在微调模式且有选中的角点按钮时才响应方向键
            if (controls.adjustmentMode && controls.activePoint >= 0) {
                const pointIndex = controls.activePoint;
                
                // 移动步长（按住Shift键时移动更大步长）
                const step = e.shiftKey ? 10 : 1;
                
                // 根据按键调整坐标
                switch (e.key) {
                    case "ArrowUp":
                        controls.points[pointIndex].y -= step;
                        break;
                    case "ArrowDown":
                        controls.points[pointIndex].y += step;
                        break;
                    case "ArrowLeft":
                        controls.points[pointIndex].x -= step;
                        break;
                    case "ArrowRight":
                        controls.points[pointIndex].x += step;
                        break;
                }
                
                // 重绘
                drawForegroundAndControls();
            }
        }
    });
    
    // 初始化图像和控制点
    initImageAndControls();
    
    return modal;
}

// 注册节点扩展
app.registerExtension({
    name: "wjb511.DistortionTransform",
    async setup() {
        // 监听图像变换更新事件
        api.addEventListener("popup_blender_update", ({ detail }) => {
            const { node_id, background_image, foreground_image } = detail;
            createPopupBlenderDialog(node_id, background_image, foreground_image);
        });
    }
}); 