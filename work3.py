import taichi as ti
import numpy as np

# 任务 1：初始化与显存预分配
ti.init(arch=ti.gpu)

RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 分配 GPU 缓冲区 (Fields)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
# 额外分配一个用于绘制控制多边形（辅助灰线）的对象池
gui_lines = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS - 1) * 2)

# 任务 2：实现 De Casteljau 算法 (纯 Python CPU 执行)
def de_casteljau(points, t):
    """
    基于给定的控制点 points 和参数 t，计算并返回贝塞尔曲线上的坐标
    """
    if not points:
        return [0.0, 0.0]
    
    # 深拷贝一份，防止修改原始的控制点列表
    pts = [list(p) for p in points]
    n = len(pts)
    
    for j in range(1, n):
        for i in range(n - j):
            pts[i][0] = (1 - t) * pts[i][0] + t * pts[i + 1][0]
            pts[i][1] = (1 - t) * pts[i][1] + t * pts[i + 1][1]
            
    return pts[0]

# 任务 3：编写 GPU 绘制内核 (Kernel)
@ti.kernel
def clear_pixels():
    # 每次重绘前清空显存（涂黑）
    for i, j in pixels:
        pixels[i, j] = [0.0, 0.0, 0.0]

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    # 在 GPU 上极速并行点亮像素
    for i in range(n):
        pos = curve_points_field[i]
        # 坐标映射：[0, 1] -> [0, RES)
        x = ti.cast(pos[0] * RES, ti.i32)
        y = ti.cast(pos[1] * RES, ti.i32)
        
        # 越界检查（关键！防止显存越界导致崩溃）
        if 0 <= x < RES and 0 <= y < RES:
            pixels[x, y] = [0.0, 1.0, 0.0]  # 赋绿色

def main():
    # 任务 4 & 5：主循环、曲线逻辑、交互响应与对象池技巧
    window = ti.ui.Window("Bezier Curve Rendering", (RES, RES))
    canvas = window.get_canvas()
    
    control_points = []
    
    # 预先分配供对象池使用的 NumPy 数组
    curve_pts_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    gui_lines_np = np.full(((MAX_CONTROL_POINTS - 1) * 2, 2), -10.0, dtype=np.float32)

    while window.running:
        # 1. 处理交互事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:  # 鼠标左键点击
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append([pos[0], pos[1]])
            elif e.key == 'c':      # 键盘 C 键清空
                control_points.clear()
                
        # 2. 清空上一帧的画面
        clear_pixels()

        # 3. 计算与绘制逻辑
        if len(control_points) >= 2:
            # CPU 端 Batching 计算 1001 个点
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                curve_pts_np[i] = de_casteljau(control_points, t)
            
            # 一次性拷贝到 GPU 并调用 Kernel 进行绘制
            curve_points_field.from_numpy(curve_pts_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        # 4. 将背景显存传递给 Canvas
        canvas.set_image(pixels)

        # 5. 更新并绘制控制点 & 辅助线 (利用对象池技巧)
        if len(control_points) > 0:
            # 刷新数组并将其“藏”在屏幕外（全部设为 -10.0）
            gui_points_np.fill(-10.0)
            gui_lines_np.fill(-10.0)
            
            # 将真实的控制点覆盖到前面
            for i, pt in enumerate(control_points):
                gui_points_np[i] = pt
                
            # 生成连线 (用于控制多边形)
            if len(control_points) >= 2:
                for i in range(len(control_points) - 1):
                    gui_lines_np[2 * i] = control_points[i]
                    gui_lines_np[2 * i + 1] = control_points[i + 1]
            
            # 将组装好的固定长度数组发送给 GPU UI 管线
            gui_points.from_numpy(gui_points_np)
            gui_lines.from_numpy(gui_lines_np)
            
            # 在 UI 层绘制
            canvas.lines(gui_lines, width=0.005, color=(0.5, 0.5, 0.5))  # 灰线
            canvas.circles(gui_points, radius=0.008, color=(1.0, 0.0, 0.0))  # 红点

        window.show()

if __name__ == "__main__":
    main()