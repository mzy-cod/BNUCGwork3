import taichi as ti
import numpy as np

# 初始化与显存预分配
ti.init(arch=ti.gpu)

RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 分配 GPU 缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_lines = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS - 1) * 2)

# CPU 端：De Casteljau 算法
def de_casteljau(points, t):
    if not points:
        return [0.0, 0.0]
    
    pts = [list(p) for p in points]
    n = len(pts)
    
    for j in range(1, n):
        for i in range(n - j):
            pts[i][0] = (1 - t) * pts[i][0] + t * pts[i + 1][0]
            pts[i][1] = (1 - t) * pts[i][1] + t * pts[i + 1][1]
            
    return pts[0]

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = [0.0, 0.0, 0.0]

# =========================================================
# 改进版 GPU 绘制内核：3x3 邻域距离衰减抗锯齿
# =========================================================
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pos = curve_points_field[i]
        
        # 1. 获取精确的亚像素浮点坐标
        cx = pos[0] * RES
        cy = pos[1] * RES
        
        # 2. 基础整数坐标
        base_x = ti.cast(cx, ti.i32)
        base_y = ti.cast(cy, ti.i32)
        
        # 3. 考察 3x3 的局部像素邻域
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                px = base_x + dx
                py = base_y + dy
                
                # 越界检查
                if 0 <= px < RES and 0 <= py < RES:
                    # 计算当前像素中心 (px, py) 与精确浮点坐标 (cx, cy) 之间的几何距离
                    dist = ti.math.sqrt((px - cx)**2 + (py - cy)**2)
                    
                    # 定义抗锯齿的有效扩散半径（例如 1.5 个像素宽度）
                    radius = 1.5
                    
                    # 如果像素落在有效半径内，则计算距离衰减权重
                    if dist < radius:
                        # 线性衰减模型：距离越近，权重越接近 1.0；距离越远，越接近 0.0
                        weight = 1.0 - (dist / radius)
                        
                        # 4. 色彩混合：使用 ti.atomic_max 防止曲线重叠区域亮度过曝
                        # 这里我们只修改 G 通道（绿色 [0, 1, 0]）
                        ti.atomic_max(pixels[px, py][1], weight)


def main():
    window = ti.ui.Window("Anti-Aliased Bezier Curve", (RES, RES))
    canvas = window.get_canvas()
    
    control_points = []
    
    # 预分配对象池
    curve_pts_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    gui_lines_np = np.full(((MAX_CONTROL_POINTS - 1) * 2, 2), -10.0, dtype=np.float32)

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append([pos[0], pos[1]])
            elif e.key == 'c':
                control_points.clear()
                
        clear_pixels()

        if len(control_points) >= 2:
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                curve_pts_np[i] = de_casteljau(control_points, t)
            
            curve_points_field.from_numpy(curve_pts_np)
            # 启动抗锯齿绘制内核
            draw_curve_kernel(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)

        if len(control_points) > 0:
            gui_points_np.fill(-10.0)
            gui_lines_np.fill(-10.0)
            
            for i, pt in enumerate(control_points):
                gui_points_np[i] = pt
                
            if len(control_points) >= 2:
                for i in range(len(control_points) - 1):
                    gui_lines_np[2 * i] = control_points[i]
                    gui_lines_np[2 * i + 1] = control_points[i + 1]
            
            gui_points.from_numpy(gui_points_np)
            gui_lines.from_numpy(gui_lines_np)
            
            canvas.lines(gui_lines, width=0.005, color=(0.5, 0.5, 0.5))
            canvas.circles(gui_points, radius=0.008, color=(1.0, 0.0, 0.0))

        window.show()

if __name__ == "__main__":
    main()