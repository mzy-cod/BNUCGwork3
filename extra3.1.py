import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 分配 GPU 缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_lines = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS - 1) * 2)

# =========================================================
# 算法 1：De Casteljau 算法 (用于贝塞尔曲线)
# =========================================================
def de_casteljau(points, t):
    if not points: return [0.0, 0.0]
    pts = [list(p) for p in points]
    n = len(pts)
    for j in range(1, n):
        for i in range(n - j):
            pts[i][0] = (1 - t) * pts[i][0] + t * pts[i + 1][0]
            pts[i][1] = (1 - t) * pts[i][1] + t * pts[i + 1][1]
    return pts[0]

# =========================================================
# 算法 2：均匀三次 B 样条矩阵求值
# =========================================================
def compute_bspline(points, total_samples=1001):
    """
    使用矩阵形式计算均匀三次 B 样条曲线。
    4个点构成1段曲线；n个点构成 n-3 段。
    """
    result = np.zeros((total_samples, 2), dtype=np.float32)
    n = len(points)
    
    if n < 4:
        return result, 0  # 点数不足4个，无法生成三次 B 样条
        
    pts = np.array(points, dtype=np.float32)
    num_segments = n - 3
    
    # 均匀三次 B 样条基矩阵 (Basis Matrix)
    M = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  0,  3,  0],
        [ 1,  4,  1,  0]
    ], dtype=np.float32) / 6.0

    for i in range(total_samples):
        # 将采样进度映射到全局参数 t，范围 [0, num_segments]
        global_t = i / (total_samples - 1) * num_segments
        
        # 确定当前点落在哪个分段上，以及在该段内的局部参数 local_t [0, 1]
        seg_idx = int(global_t)
        if seg_idx >= num_segments:
            seg_idx = num_segments - 1
            
        local_t = global_t - seg_idx
        if i == total_samples - 1:
            local_t = 1.0  # 确保最后一个点精确抵达参数末端
            
        T = np.array([local_t**3, local_t**2, local_t, 1.0], dtype=np.float32)
        
        # 提取当前分段的 4 个控制点构成的几何矩阵 G
        G = pts[seg_idx : seg_idx + 4]
        
        # 矩阵相乘：P(t) = T * M * G
        result[i] = T @ M @ G
        
    return result, total_samples

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = [0.0, 0.0, 0.0]

# =========================================================
# 修改版 GPU 渲染内核：支持外部传入 RGB 颜色
# =========================================================
@ti.kernel
def draw_curve_kernel(n: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    for i in range(n):
        pos = curve_points_field[i]
        cx = pos[0] * RES
        cy = pos[1] * RES
        base_x = ti.cast(cx, ti.i32)
        base_y = ti.cast(cy, ti.i32)
        
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                px = base_x + dx
                py = base_y + dy
                if 0 <= px < RES and 0 <= py < RES:
                    dist = ti.math.sqrt((px - cx)**2 + (py - cy)**2)
                    radius = 1.5
                    if dist < radius:
                        weight = 1.0 - (dist / radius)
                        ti.atomic_max(pixels[px, py][0], weight * r)
                        ti.atomic_max(pixels[px, py][1], weight * g)
                        ti.atomic_max(pixels[px, py][2], weight * b)

def main():
    window = ti.ui.Window("Bezier vs B-Spline (Press 'B' to switch)", (RES, RES))
    canvas = window.get_canvas()
    
    control_points = []
    mode = 'bezier'  # 初始模式为贝塞尔曲线
    
    curve_pts_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    gui_lines_np = np.full(((MAX_CONTROL_POINTS - 1) * 2, 2), -10.0, dtype=np.float32)

    print("已启动渲染器。当前模式: Bezier (绿色)。点击添加控制点。")

    while window.running:
        # 1. 处理交互事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append([pos[0], pos[1]])
            elif e.key == 'c':
                control_points.clear()
            elif e.key == 'b':
                mode = 'bspline' if mode == 'bezier' else 'bezier'
                print(f"模式切换！当前模式: {'B-Spline (蓝色)' if mode == 'bspline' else 'Bezier (绿色)'}")
                
        clear_pixels()

        # 2. 核心计算与数据拷贝逻辑
        points_to_draw = 0
        r, g, b_color = 0.0, 1.0, 0.0 # 默认绿色

        if mode == 'bezier' and len(control_points) >= 2:
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                curve_pts_np[i] = de_casteljau(control_points, t)
            points_to_draw = NUM_SEGMENTS + 1
            r, g, b_color = 0.0, 1.0, 0.0  # 贝塞尔使用绿色
            
        elif mode == 'bspline':
            pts, count = compute_bspline(control_points, NUM_SEGMENTS + 1)
            if count > 0:
                curve_pts_np[:count] = pts[:count]
                points_to_draw = count
                r, g, b_color = 0.0, 0.5, 1.0  # B样条使用深天蓝色

        # 3. 如果有点需要画，交给 GPU
        if points_to_draw > 0:
            curve_points_field.from_numpy(curve_pts_np)
            draw_curve_kernel(points_to_draw, r, g, b_color)

        canvas.set_image(pixels)

        # 4. 渲染控制多边形及控制点
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