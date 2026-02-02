"""
多针细胞染料灌注调度问题 - 可视化模块
==================================================
参考 C++ ImPlot 实现，使用 matplotlib 进行动态可视化
支持:
- 细胞分布图 (按颜色区分，已处理/待处理状态)
- 针的位置和方向 (针身线段)
- 时间轴动画播放
- 时间线图 (各颜色处理进度)
- 碰撞检测显示
- 交互控制 (播放/暂停/速度调节)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import platform

# 设置中文字体支持
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 从 main.py 导入必要的类
from main import (
    Cell, Needle, CellTask, CellInjectionProblem, ProblemConfig,
    CollisionDetectionMode, create_test_instance, solve_cell_injection,
    ScheduleDecoder
)


# =============================================================================
# 颜色配置
# =============================================================================

# 颜色调色板 (与 C++ 代码一致)
COLOR_PALETTE = {
    0: '#FF0000',  # 红色
    1: '#00FF00',  # 绿色
    2: '#0000FF',  # 蓝色
    3: '#FFFF00',  # 黄色
    4: '#FF00FF',  # 品红
    5: '#00FFFF',  # 青色
    6: '#FF8000',  # 橙色
    7: '#8000FF',  # 紫色
}

def get_color(color_id: int, alpha: float = 1.0) -> str:
    """获取颜色，支持透明度"""
    base_color = COLOR_PALETTE.get(color_id % len(COLOR_PALETTE), '#FFFFFF')
    if alpha < 1.0:
        # 转换为 RGBA
        r = int(base_color[1:3], 16) / 255
        g = int(base_color[3:5], 16) / 255
        b = int(base_color[5:7], 16) / 255
        return (r, g, b, alpha)
    return base_color


# =============================================================================
# 仿真状态快照
# =============================================================================

@dataclass
class SimulationSnapshot:
    """仿真状态快照"""
    time: float                              # 当前时间
    needle_positions: Dict[int, Tuple[float, float, float]]  # 针ID -> (x, y, z)
    needle_targets: Dict[int, Optional[int]]  # 针ID -> 目标细胞ID (None表示空闲)
    needle_states: Dict[int, str]            # 针ID -> 状态 ("idle", "find", "wait")
    colored_cells: set                       # 已完成着色的细胞ID集合
    active_cells: Dict[int, int]             # 正在处理的细胞ID -> 针ID
    

def generate_snapshots(
    tasks: List[CellTask], 
    problem: CellInjectionProblem,
    time_resolution: float = 1.0
) -> List[SimulationSnapshot]:
    """
    根据任务列表生成仿真状态快照序列
    
    Args:
        tasks: 任务列表
        problem: 问题实例
        time_resolution: 时间分辨率 (秒)
        
    Returns:
        快照列表
    """
    if not tasks:
        return []
    
    config = problem.config
    makespan = max(t.end_time for t in tasks)
    
    # 生成时间点
    times = np.arange(0, makespan + time_resolution, time_resolution)
    
    snapshots = []
    
    for t in times:
        # 初始化针状态
        needle_positions = {}
        needle_targets = {}
        needle_states = {}
        colored_cells = set()
        active_cells = {}
        
        for needle in problem.needles:
            # 默认在初始位置
            needle_positions[needle.id] = (needle.init_x, needle.init_y, needle.init_z)
            needle_targets[needle.id] = None
            needle_states[needle.id] = "idle"
        
        # 根据任务更新状态
        for task in tasks:
            cell = problem.cells[task.cell_id]
            
            if task.end_time <= t:
                # 任务已完成
                colored_cells.add(task.cell_id)
            elif task.start_time <= t < task.end_time:
                # 任务正在进行
                needle_positions[task.needle_id] = (cell.x, cell.y, cell.z)
                needle_targets[task.needle_id] = task.cell_id
                active_cells[task.cell_id] = task.needle_id
                
                if t < task.find_end_time:
                    needle_states[task.needle_id] = "find"
                else:
                    needle_states[task.needle_id] = "wait"
        
        snapshots.append(SimulationSnapshot(
            time=t,
            needle_positions=needle_positions,
            needle_targets=needle_targets,
            needle_states=needle_states,
            colored_cells=colored_cells,
            active_cells=active_cells
        ))
    
    return snapshots


# =============================================================================
# 可视化器类
# =============================================================================

class ScheduleVisualizer:
    """调度可视化器"""
    
    def __init__(
        self, 
        tasks: List[CellTask], 
        problem: CellInjectionProblem,
        time_resolution: float = 1.0
    ):
        """
        初始化可视化器
        
        Args:
            tasks: 任务列表
            problem: 问题实例
            time_resolution: 时间分辨率 (秒)
        """
        self.tasks = tasks
        self.problem = problem
        self.config = problem.config
        self.time_resolution = time_resolution
        
        # 生成快照
        self.snapshots = generate_snapshots(tasks, problem, time_resolution)
        self.n_frames = len(self.snapshots)
        self.current_frame = 0
        
        # 计算边界
        cell_xs = [c.x for c in problem.cells]
        cell_ys = [c.y for c in problem.cells]
        self.x_min = min(cell_xs) - 100
        self.x_max = max(cell_xs) + 100
        self.y_min = min(cell_ys) - 100
        self.y_max = max(cell_ys) + 100
        
        # 计算makespan
        self.makespan = max(t.end_time for t in tasks) if tasks else 0
        
        # 动画状态
        self.animating = False
        self.animation_speed = 1.0
        self.show_needle_body = True
        self.show_collision_zones = False
        
        # 按颜色分组任务，计算累积进度
        self.tasks_by_color = {}
        for task in tasks:
            if task.color not in self.tasks_by_color:
                self.tasks_by_color[task.color] = []
            self.tasks_by_color[task.color].append(task)
        
        # 初始化图形
        self._init_figure()
    
    def _init_figure(self):
        """初始化图形"""
        # 创建图形和子图
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('多针细胞染料灌注调度可视化', fontsize=14, fontweight='bold')
        
        # 布局: 左侧细胞分布图，右上时间线，右下控制面板
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[2, 1, 0.5],
                                   hspace=0.3, wspace=0.2)
        
        # 细胞分布图
        self.ax_cells = self.fig.add_subplot(gs[:2, 0])
        self.ax_cells.set_xlabel('X (μm)')
        self.ax_cells.set_ylabel('Y (μm)')
        self.ax_cells.set_xlim(self.x_min, self.x_max)
        self.ax_cells.set_ylim(self.y_min, self.y_max)
        self.ax_cells.set_aspect('equal')
        self.ax_cells.grid(True, alpha=0.3)
        
        # 时间线图
        self.ax_timeline = self.fig.add_subplot(gs[0, 1])
        self.ax_timeline.set_xlabel('时间 (秒)')
        self.ax_timeline.set_ylabel('已完成细胞数')
        self.ax_timeline.set_xlim(0, self.makespan + 10)
        self.ax_timeline.grid(True, alpha=0.3)
        
        # 状态信息区
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_info.axis('off')
        
        # 控制面板区域
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')
        
        # 初始化绑定对象
        self._init_bindable_objects()
        
        # 添加控件
        self._add_controls()
        
        # 绘制初始帧
        self._draw_frame(0)
    
    def _init_bindable_objects(self):
        """初始化可更新的图形对象"""
        # 细胞散点图
        self.pending_scatters = {}  # 颜色 -> scatter对象 (待处理)
        self.done_scatters = {}     # 颜色 -> scatter对象 (已完成)
        self.active_scatters = {}   # 颜色 -> scatter对象 (正在处理)
        
        # 针的图形对象
        self.needle_lines = {}      # 针ID -> Line2D对象 (针身)
        self.needle_markers = {}    # 针ID -> scatter对象 (针尖)
        
        # 时间线
        self.time_line = None
        self.progress_lines = {}    # 颜色 -> Line2D对象
        
        # 文本
        self.time_text = None
        self.info_text = None
    
    def _add_controls(self):
        """添加控制控件"""
        # 时间滑块
        ax_slider = plt.axes([0.15, 0.08, 0.55, 0.03])
        self.time_slider = Slider(
            ax_slider, '时间', 0, self.makespan, 
            valinit=0, valstep=self.time_resolution
        )
        self.time_slider.on_changed(self._on_slider_change)
        
        # 播放/暂停按钮
        ax_play = plt.axes([0.75, 0.08, 0.08, 0.04])
        self.play_button = Button(ax_play, '▶ 播放')
        self.play_button.on_clicked(self._on_play_click)
        
        # 重置按钮
        ax_reset = plt.axes([0.85, 0.08, 0.08, 0.04])
        self.reset_button = Button(ax_reset, '⟲ 重置')
        self.reset_button.on_clicked(self._on_reset_click)
        
        # 速度滑块
        ax_speed = plt.axes([0.15, 0.03, 0.25, 0.03])
        self.speed_slider = Slider(
            ax_speed, '速度', 0.1, 5.0, 
            valinit=1.0, valstep=0.1
        )
        self.speed_slider.on_changed(self._on_speed_change)
        
        # 选项复选框
        ax_check = plt.axes([0.5, 0.01, 0.2, 0.06])
        self.check_buttons = CheckButtons(
            ax_check, 
            ['显示针身', '显示碰撞区'],
            [True, False]
        )
        self.check_buttons.on_clicked(self._on_check_click)
    
    def _on_slider_change(self, val):
        """时间滑块变化回调"""
        frame = int(val / self.time_resolution)
        frame = min(frame, self.n_frames - 1)
        self.current_frame = frame
        self._draw_frame(frame)
        self.fig.canvas.draw_idle()
    
    def _on_play_click(self, event):
        """播放按钮点击回调"""
        self.animating = not self.animating
        if self.animating:
            self.play_button.label.set_text('⏸ 暂停')
            self._start_animation()
        else:
            self.play_button.label.set_text('▶ 播放')
    
    def _on_reset_click(self, event):
        """重置按钮点击回调"""
        self.animating = False
        self.play_button.label.set_text('▶ 播放')
        self.current_frame = 0
        self.time_slider.set_val(0)
        self._draw_frame(0)
        self.fig.canvas.draw_idle()
    
    def _on_speed_change(self, val):
        """速度滑块变化回调"""
        self.animation_speed = val
    
    def _on_check_click(self, label):
        """复选框点击回调"""
        if label == '显示针身':
            self.show_needle_body = not self.show_needle_body
        elif label == '显示碰撞区':
            self.show_collision_zones = not self.show_collision_zones
        self._draw_frame(self.current_frame)
        self.fig.canvas.draw_idle()
    
    def _start_animation(self):
        """开始动画"""
        def animate(frame):
            if not self.animating:
                return
            
            self.current_frame += 1
            if self.current_frame >= self.n_frames:
                self.current_frame = 0
            
            # 更新滑块（不触发回调）
            self.time_slider.set_val(self.current_frame * self.time_resolution)
        
        # 计算间隔时间 (毫秒)
        interval = int(self.time_resolution * 1000 / self.animation_speed)
        self.anim = animation.FuncAnimation(
            self.fig, animate, 
            frames=self.n_frames,
            interval=interval,
            repeat=True
        )
        self.fig.canvas.draw_idle()
    
    def _draw_frame(self, frame_idx: int):
        """绘制指定帧"""
        if frame_idx >= len(self.snapshots):
            return
        
        snapshot = self.snapshots[frame_idx]
        
        # 清除细胞图
        self.ax_cells.clear()
        self.ax_cells.set_xlabel('X (μm)')
        self.ax_cells.set_ylabel('Y (μm)')
        self.ax_cells.set_xlim(self.x_min, self.x_max)
        self.ax_cells.set_ylim(self.y_min, self.y_max)
        self.ax_cells.set_aspect('equal')
        self.ax_cells.grid(True, alpha=0.3)
        self.ax_cells.set_title(f'时间: {snapshot.time:.1f}s ({snapshot.time/60:.1f}min)')
        
        # 绘制细胞
        self._draw_cells(snapshot)
        
        # 绘制针
        self._draw_needles(snapshot)
        
        # 绘制碰撞区（如果启用）
        if self.show_collision_zones:
            self._draw_collision_zones(snapshot)
        
        # 添加图例
        self._add_legend()
        
        # 更新时间线图
        self._draw_timeline(snapshot)
        
        # 更新状态信息
        self._draw_info(snapshot)
    
    def _draw_cells(self, snapshot: SimulationSnapshot):
        """绘制细胞"""
        for color in self.problem.cells_by_color.keys():
            pending_x, pending_y = [], []
            done_x, done_y = [], []
            active_x, active_y = [], []
            
            for cell_id in self.problem.cells_by_color[color]:
                cell = self.problem.cells[cell_id]
                
                if cell_id in snapshot.colored_cells:
                    done_x.append(cell.x)
                    done_y.append(cell.y)
                elif cell_id in snapshot.active_cells:
                    active_x.append(cell.x)
                    active_y.append(cell.y)
                else:
                    pending_x.append(cell.x)
                    pending_y.append(cell.y)
            
            base_color = get_color(color)
            
            # 待处理细胞（空心，半透明）
            if pending_x:
                self.ax_cells.scatter(
                    pending_x, pending_y, 
                    c='none', edgecolors=base_color, 
                    s=60, linewidths=1.5, alpha=0.5,
                    label=f'颜色{color} 待处理' if color == 0 else ''
                )
            
            # 已完成细胞（实心）
            if done_x:
                self.ax_cells.scatter(
                    done_x, done_y, 
                    c=base_color, edgecolors='black',
                    s=80, linewidths=0.5, alpha=0.9,
                    label=f'颜色{color} 已完成' if color == 0 else ''
                )
            
            # 正在处理的细胞（实心，带高亮边框）
            if active_x:
                self.ax_cells.scatter(
                    active_x, active_y, 
                    c=base_color, edgecolors='white',
                    s=120, linewidths=3, alpha=1.0,
                    marker='*'
                )
    
    def _draw_needles(self, snapshot: SimulationSnapshot):
        """绘制针"""
        for needle in self.problem.needles:
            pos = snapshot.needle_positions[needle.id]
            state = snapshot.needle_states[needle.id]
            color = get_color(needle.color)
            
            # 针尖位置
            tip_x, tip_y = pos[0], pos[1]
            
            # 计算针座位置
            if self.config.collision_mode == CollisionDetectionMode.MODE_2D:
                direction = np.array([np.cos(needle.theta_xy), np.sin(needle.theta_xy)])
            else:
                cos_z = np.cos(needle.theta_z)
                direction = np.array([
                    cos_z * np.cos(needle.theta_xy),
                    cos_z * np.sin(needle.theta_xy)
                ])
            
            base_x = tip_x + needle.length * direction[0]
            base_y = tip_y + needle.length * direction[1]
            
            # 根据状态设置样式
            if state == "find":
                marker_size = 200
                line_width = 3
                alpha = 1.0
            elif state == "wait":
                marker_size = 150
                line_width = 2
                alpha = 0.8
            else:  # idle
                marker_size = 100
                line_width = 1.5
                alpha = 0.5
            
            # 绘制针身（如果启用）
            if self.show_needle_body:
                self.ax_cells.plot(
                    [tip_x, base_x], [tip_y, base_y],
                    color=color, linewidth=line_width, alpha=alpha,
                    solid_capstyle='round'
                )
            
            # 绘制针尖（菱形标记）
            self.ax_cells.scatter(
                [tip_x], [tip_y],
                c=color, marker='D', s=marker_size,
                edgecolors='black', linewidths=1,
                alpha=alpha, zorder=10
            )
            
            # 添加针编号标签
            self.ax_cells.annotate(
                f'N{needle.id}',
                (tip_x, tip_y),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold',
                color=color
            )
    
    def _draw_collision_zones(self, snapshot: SimulationSnapshot):
        """绘制碰撞检测区域"""
        # 绘制正在处理的细胞的安全距离圈
        for cell_id, needle_id in snapshot.active_cells.items():
            cell = self.problem.cells[cell_id]
            
            # 针尖安全距离圈
            circle_tip = plt.Circle(
                (cell.x, cell.y), self.config.d_tip,
                fill=False, color='red', linestyle='--', alpha=0.5, linewidth=1
            )
            self.ax_cells.add_patch(circle_tip)
    
    def _add_legend(self):
        """添加图例"""
        legend_elements = []
        
        for color in sorted(self.problem.cells_by_color.keys()):
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=get_color(color), markersize=8,
                       label=f'颜色 {color}')
            )
        
        legend_elements.extend([
            Line2D([0], [0], marker='D', color='w', 
                   markerfacecolor='gray', markersize=8,
                   label='针尖位置'),
            Line2D([0], [0], marker='*', color='w', 
                   markerfacecolor='yellow', markersize=10,
                   markeredgecolor='black',
                   label='正在处理')
        ])
        
        self.ax_cells.legend(
            handles=legend_elements, 
            loc='upper right',
            fontsize=8,
            framealpha=0.9
        )
    
    def _draw_timeline(self, snapshot: SimulationSnapshot):
        """绘制时间线图"""
        self.ax_timeline.clear()
        self.ax_timeline.set_xlabel('时间 (秒)')
        self.ax_timeline.set_ylabel('已完成细胞数')
        self.ax_timeline.set_xlim(0, self.makespan + 10)
        self.ax_timeline.grid(True, alpha=0.3)
        self.ax_timeline.set_title('处理进度')
        
        # 为每种颜色绘制累积曲线
        for color, tasks_list in self.tasks_by_color.items():
            # 按完成时间排序
            sorted_tasks = sorted(tasks_list, key=lambda t: t.end_time)
            
            times = [0]
            counts = [0]
            count = 0
            
            for task in sorted_tasks:
                count += 1
                times.append(task.end_time)
                counts.append(count)
            
            # 延续到当前时间
            times.append(snapshot.time)
            # 计算当前时间的完成数
            current_count = sum(1 for t in sorted_tasks if t.end_time <= snapshot.time)
            counts.append(current_count)
            
            self.ax_timeline.plot(
                times, counts,
                color=get_color(color), linewidth=2,
                label=f'颜色 {color}'
            )
        
        # 绘制当前时间线
        self.ax_timeline.axvline(
            x=snapshot.time, color='black', 
            linestyle='--', linewidth=1.5, alpha=0.7
        )
        
        self.ax_timeline.legend(loc='upper left', fontsize=8)
        
        # 设置Y轴范围
        max_cells = max(len(cells) for cells in self.problem.cells_by_color.values())
        self.ax_timeline.set_ylim(0, max_cells + 1)
    
    def _draw_info(self, snapshot: SimulationSnapshot):
        """绘制状态信息"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # 统计信息
        total_colored = len(snapshot.colored_cells)
        total_cells = self.problem.n_cells
        progress = total_colored / total_cells * 100
        
        # 各针状态
        needle_info = []
        for needle in self.problem.needles:
            state = snapshot.needle_states[needle.id]
            target = snapshot.needle_targets[needle.id]
            state_cn = {'idle': 'Idle', 'find': 'Find', 'wait': 'Wait'}[state]
            if target is not None:
                needle_info.append(f"N{needle.id}(C{needle.color}): {state_cn} -> Cell{target}")
            else:
                needle_info.append(f"N{needle.id}(C{needle.color}): {state_cn}")
        
        info_text = f"""仿真状态
Time: {snapshot.time:.1f}s ({snapshot.time/60:.1f}min)
Progress: {total_colored}/{total_cells} ({progress:.1f}%)
各针状态:
""" + '\n'.join(f"  {info}" for info in needle_info)
        
        self.ax_info.text(
            0.05, 0.95, info_text,
            transform=self.ax_info.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    def show(self):
        """显示可视化窗口"""
        plt.tight_layout()
        plt.show()
    
    def save_animation(self, filename: str, fps: int = 30):
        """
        保存动画到文件
        
        Args:
            filename: 输出文件名 (支持 .mp4, .gif)
            fps: 帧率
        """
        def update(frame):
            self._draw_frame(frame)
            return []
        
        anim = animation.FuncAnimation(
            self.fig, update,
            frames=self.n_frames,
            interval=1000/fps,
            blit=False
        )
        
        print(f"正在保存动画到 {filename}...")
        anim.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg', fps=fps)
        print("保存完成!")


# =============================================================================
# 静态可视化函数
# =============================================================================

def plot_schedule_gantt(tasks: List[CellTask], problem: CellInjectionProblem):
    """
    绘制甘特图
    
    显示各针的任务时间线
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 按针分组任务
    by_needle = {}
    for task in tasks:
        if task.needle_id not in by_needle:
            by_needle[task.needle_id] = []
        by_needle[task.needle_id].append(task)
    
    y_ticks = []
    y_labels = []
    
    for i, (needle_id, needle_tasks) in enumerate(sorted(by_needle.items())):
        y_pos = i * 2
        y_ticks.append(y_pos)
        y_labels.append(f'针 {needle_id}\n(颜色 {problem.needle_to_color[needle_id]})')
        
        for task in needle_tasks:
            color = get_color(task.color)
            
            # Find阶段 (深色)
            ax.barh(
                y_pos, task.find_end_time - task.start_time,
                left=task.start_time, height=0.8,
                color=color, edgecolor='black', linewidth=0.5,
                alpha=1.0
            )
            
            # Wait阶段 (浅色)
            ax.barh(
                y_pos, task.end_time - task.find_end_time,
                left=task.find_end_time, height=0.8,
                color=color, edgecolor='black', linewidth=0.5,
                alpha=0.5
            )
            
            # 细胞ID标签
            ax.text(
                (task.start_time + task.end_time) / 2, y_pos,
                f'{task.cell_id}',
                ha='center', va='center',
                fontsize=7, fontweight='bold'
            )
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('时间 (秒)')
    ax.set_title('调度甘特图 (深色=Find阶段, 浅色=Wait阶段)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=get_color(c), edgecolor='black', label=f'颜色 {c}')
        for c in sorted(problem.cells_by_color.keys())
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_cell_distribution(problem: CellInjectionProblem, tasks: List[CellTask] = None):
    """
    绘制细胞分布静态图
    
    显示细胞位置和针的初始位置
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制细胞
    for color, cell_ids in problem.cells_by_color.items():
        xs = [problem.cells[cid].x for cid in cell_ids]
        ys = [problem.cells[cid].y for cid in cell_ids]
        
        ax.scatter(
            xs, ys, c=get_color(color),
            s=80, edgecolors='black', linewidths=0.5,
            label=f'颜色 {color} ({len(cell_ids)}个)',
            alpha=0.8
        )
        
        # 添加细胞ID标签
        for cid in cell_ids:
            cell = problem.cells[cid]
            ax.annotate(
                str(cid), (cell.x, cell.y),
                xytext=(2, 2), textcoords='offset points',
                fontsize=6, alpha=0.7
            )
    
    # 绘制针的初始位置和方向
    for needle in problem.needles:
        color = get_color(needle.color)
        
        # 针尖（初始位置）
        ax.scatter(
            [needle.init_x], [needle.init_y],
            c=color, marker='D', s=150,
            edgecolors='black', linewidths=2,
            zorder=10
        )
        
        # 针身方向
        if problem.config.collision_mode == CollisionDetectionMode.MODE_2D:
            dx = np.cos(needle.theta_xy) * needle.length
            dy = np.sin(needle.theta_xy) * needle.length
        else:
            cos_z = np.cos(needle.theta_z)
            dx = cos_z * np.cos(needle.theta_xy) * needle.length
            dy = cos_z * np.sin(needle.theta_xy) * needle.length
        
        ax.arrow(
            needle.init_x, needle.init_y, dx, dy,
            head_width=20, head_length=10,
            fc=color, ec='black', linewidth=1,
            alpha=0.7
        )
        
        ax.annotate(
            f'针{needle.id}',
            (needle.init_x, needle.init_y),
            xytext=(10, 10), textcoords='offset points',
            fontsize=10, fontweight='bold',
            color=color
        )
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('细胞分布与针初始位置')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pareto_front(F: np.ndarray, highlight_idx: int = None):
    """
    绘制Pareto前沿
    
    Args:
        F: 目标函数值矩阵 (n_solutions x n_objectives)
        highlight_idx: 高亮显示的解索引
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    objectives = [
        ('Makespan (秒)', 0),
        ('总移动距离 (μm)', 1),
        ('长距离移动次数', 2),
        ('总空闲时间 (秒)', 3)
    ]
    
    # 绘制两两目标的散点图
    pairs = [(0, 1), (0, 2), (0, 3), (1, 3)]
    
    for ax, (i, j) in zip(axes.flat, pairs):
        ax.scatter(F[:, i], F[:, j], c='blue', alpha=0.6, s=50)
        
        if highlight_idx is not None:
            ax.scatter(
                [F[highlight_idx, i]], [F[highlight_idx, j]],
                c='red', s=150, marker='*', zorder=10,
                label='最优Makespan解'
            )
        
        ax.set_xlabel(objectives[i][0])
        ax.set_ylabel(objectives[j][0])
        ax.grid(True, alpha=0.3)
        
        if highlight_idx is not None:
            ax.legend()
    
    fig.suptitle('Pareto前沿', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# 主函数
# =============================================================================

def visualize_solution(
    tasks: List[CellTask], 
    problem: CellInjectionProblem,
    time_resolution: float = 2.0
):
    """
    可视化单个解
    
    Args:
        tasks: 任务列表
        problem: 问题实例
        time_resolution: 时间分辨率
    """
    visualizer = ScheduleVisualizer(tasks, problem, time_resolution)
    visualizer.show()


def main():
    """演示可视化功能"""
    print("=" * 70)
    print("多针细胞染料灌注调度问题 - 可视化演示")
    print("=" * 70)
    
    # 创建问题实例
    print("\n[1] 创建问题实例...")
    problem = create_test_instance(
        n_cells=20,      # 使用较少细胞以便观察
        n_colors=4,
        field_size=800,
        seed=42
    )
    
    print(f"    细胞总数: {problem.n_cells}")
    print(f"    颜色/针数量: {problem.n_colors}")
    
    # 执行优化
    print("\n[2] 执行优化...")
    X, F, schedules = solve_cell_injection(
        problem,
        pop_size=50,
        n_gen=50,
        seed=42,
        verbose=True,
        use_color_aware_operators=True
    )
    
    # 选择最优makespan解
    best_idx = np.argmin(F[:, 0])
    best_tasks = schedules[best_idx]
    
    print(f"\n[3] 最优解 (Makespan={F[best_idx, 0]:.1f}s)")
    
    # 绘制静态图
    print("\n[4] 绘制静态分析图...")
    
    # 细胞分布图
    fig1 = plot_cell_distribution(problem, best_tasks)
    
    # 甘特图
    fig2 = plot_schedule_gantt(best_tasks, problem)
    
    # Pareto前沿
    fig3 = plot_pareto_front(F, highlight_idx=best_idx)
    
    # 动态可视化
    print("\n[5] 启动动态可视化...")
    print("    - 使用滑块调整时间")
    print("    - 点击'播放'按钮开始动画")
    print("    - 调整速度滑块改变播放速度")
    
    visualizer = ScheduleVisualizer(best_tasks, problem, time_resolution=5.0)
    visualizer.show()


if __name__ == "__main__":
    main()
