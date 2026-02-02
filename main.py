"""
多针细胞染料灌注调度问题 - NSGA-II求解器
==================================================
基于pymoo库实现，完全符合数学模型中定义的:
- 4个优化目标
- 颜色-针绑定约束
- 相机互斥约束 (Find阶段)
- 碰撞避免约束
- 并行机制 (Wait阶段)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
import heapq

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback


# =============================================================================
# 几何工具函数
# =============================================================================

def point_to_segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    """
    计算点P到线段AB的最短距离
    
    Args:
        P: 点坐标 (3,)
        A: 线段起点 (3,)
        B: 线段终点 (3,)
        
    Returns:
        distance: 最短距离
        t: 最近点在线段上的参数 t ∈ [0,1], 0表示在A点, 1表示在B点
    """
    AB = B - A
    AP = P - A
    
    ab_squared = np.dot(AB, AB)
    if ab_squared < 1e-10:  # A和B重合
        return np.linalg.norm(AP), 0.0
    
    t = np.dot(AP, AB) / ab_squared
    t = np.clip(t, 0.0, 1.0)
    
    closest_point = A + t * AB
    distance = np.linalg.norm(P - closest_point)
    
    return distance, t


def segment_to_segment_distance(A1: np.ndarray, B1: np.ndarray, 
                                  A2: np.ndarray, B2: np.ndarray) -> Tuple[float, float, float]:
    """
    计算两条线段的最短距离
    线段1: A1 -> B1
    线段2: A2 -> B2
    
    Args:
        A1, B1: 线段1的端点
        A2, B2: 线段2的端点
        
    Returns:
        distance: 最短距离
        t1: 线段1上最近点的参数 t1 ∈ [0,1]
        t2: 线段2上最近点的参数 t2 ∈ [0,1]
    """
    d1 = B1 - A1  # 线段1的方向向量
    d2 = B2 - A2  # 线段2的方向向量
    r = A1 - A2
    
    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)
    
    EPSILON = 1e-10
    
    # 检查两条线段是否都退化为点
    if a < EPSILON and e < EPSILON:
        return np.linalg.norm(r), 0.0, 0.0
    
    if a < EPSILON:
        # 线段1退化为点
        t1 = 0.0
        t2 = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < EPSILON:
            # 线段2退化为点
            t2 = 0.0
            t1 = np.clip(-c / a, 0.0, 1.0)
        else:
            # 一般情况
            b = np.dot(d1, d2)
            denom = a * e - b * b
            
            if abs(denom) > EPSILON:
                t1 = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                t1 = 0.0
            
            # 计算线段2上的最近点
            t2 = (b * t1 + f) / e
            
            # 如果t2不在[0,1]范围内，需要clamp并重新计算t1
            if t2 < 0.0:
                t2 = 0.0
                t1 = np.clip(-c / a, 0.0, 1.0)
            elif t2 > 1.0:
                t2 = 1.0
                t1 = np.clip((b - c) / a, 0.0, 1.0)
    
    closest1 = A1 + t1 * d1
    closest2 = A2 + t2 * d2
    distance = np.linalg.norm(closest1 - closest2)
    
    return distance, t1, t2


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class Cell:
    """细胞数据类"""
    id: int
    x: float
    y: float
    z: float
    color: int  # 所属颜色组 (由图着色算法预先确定)


@dataclass
class Needle:
    """注射针数据类"""
    id: int
    color: int              # 对应处理的颜色 (针-颜色一一对应)
    init_x: float           # 初始位置 x
    init_y: float           # 初始位置 y
    init_z: float           # 初始位置 z
    theta_xy: float         # 水平安装角度
    theta_z: float          # 垂直安装角度
    length: float           # 针的有效长度
    
    def get_direction_vector(self) -> np.ndarray:
        """
        获取针的方向向量 (从针尖指向针座)
        d_p = (cos(θ_z)·cos(θ_xy), cos(θ_z)·sin(θ_xy), sin(θ_z))
        """
        cos_z = np.cos(self.theta_z)
        sin_z = np.sin(self.theta_z)
        cos_xy = np.cos(self.theta_xy)
        sin_xy = np.sin(self.theta_xy)
        return np.array([cos_z * cos_xy, cos_z * sin_xy, sin_z])
    
    def get_tip_position(self, cell: 'Cell') -> np.ndarray:
        """
        获取针尖位置 (处理细胞时针尖在细胞位置)
        T_p(c) = (x_c, y_c, z_c)
        """
        return np.array([cell.x, cell.y, cell.z])
    
    def get_base_position(self, cell: 'Cell') -> np.ndarray:
        """
        获取针座位置
        B_p(c) = T_p(c) + L_p · d_p
        """
        tip = self.get_tip_position(cell)
        direction = self.get_direction_vector()
        return tip + self.length * direction


class CollisionDetectionMode(Enum):
    """碰撞检测模式"""
    MODE_2D = "2D"  # 忽略Z坐标和垂直角度
    MODE_3D = "3D"  # 完整三维几何


@dataclass
class ProblemConfig:
    """问题配置参数"""
    t_find: float = 60.0        # Find阶段固定耗时 (秒), 约1分钟
    t_wait: float = 240.0       # Wait阶段耗时 (秒), 约4分钟
    velocity: float = 10.0      # 针的移动速度 (μm/s)
    d_tip: float = 50.0         # 针尖安全距离 (用于针尖-针尖、针尖-针身碰撞检测) (μm)
    d_body: float = 100.0       # 针身安全距离 (用于针身-针身碰撞检测) (μm)
    d_long: float = 500.0       # 长距离移动阈值 (μm)
    collision_mode: CollisionDetectionMode = CollisionDetectionMode.MODE_2D  # 碰撞检测模式


@dataclass
class CellTask:
    """细胞处理任务"""
    cell_id: int
    needle_id: int
    color: int
    start_time: float = 0.0         # Find阶段开始时间 (s_c)
    find_end_time: float = 0.0      # Find阶段结束时间 = Wait阶段开始时间
    end_time: float = 0.0           # 完成时间 (e_c)
    
    def __lt__(self, other):
        return self.start_time < other.start_time


# =============================================================================
# 问题数据管理类
# =============================================================================

class CellInjectionProblem:
    """
    多针细胞染料灌注调度问题的数据管理类
    
    负责:
    - 管理细胞、针的数据
    - 预计算距离矩阵和各类指示器
    - 提供便捷的查询接口
    """
    
    def __init__(self, cells: List[Cell], needles: List[Needle], config: ProblemConfig):
        """
        初始化问题实例
        
        Args:
            cells: 细胞列表
            needles: 针列表 (数量 = 颜色数量)
            config: 问题配置参数
        """
        self.cells = cells
        self.needles = needles
        self.config = config
        self.n_cells = len(cells)
        self.n_colors = len(needles)
        self.n_needles = len(needles)
        
        # 验证针-颜色一一对应
        assert self.n_needles == self.n_colors, "针的数量必须等于颜色数量"
        
        # 建立颜色到针的映射 π: P → M
        self.color_to_needle: Dict[int, int] = {n.color: n.id for n in needles}
        self.needle_to_color: Dict[int, int] = {n.id: n.color for n in needles}
        
        # 按颜色分组细胞 C_m
        self.cells_by_color: Dict[int, List[int]] = {}
        for cell in cells:
            if cell.color not in self.cells_by_color:
                self.cells_by_color[cell.color] = []
            self.cells_by_color[cell.color].append(cell.id)
        
        # 细胞ID到索引的映射
        self.cell_id_to_idx = {c.id: i for i, c in enumerate(cells)}
        
        # 预计算各类矩阵
        self._precompute_distances()
        self._precompute_indicators()
    
    def _precompute_distances(self):
        """预计算所有距离 (d_cc', d_0c^p)"""
        n = self.n_cells
        
        # 细胞间欧氏距离矩阵 d_cc'
        self.dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    ci, cj = self.cells[i], self.cells[j]
                    self.dist_matrix[i, j] = np.sqrt(
                        (ci.x - cj.x)**2 + (ci.y - cj.y)**2 + (ci.z - cj.z)**2
                    )
        
        # 针初始位置到各细胞的距离 d_0c^p
        self.init_dist = np.zeros((self.n_needles, n))
        for needle in self.needles:
            for cell in self.cells:
                self.init_dist[needle.id, cell.id] = np.sqrt(
                    (needle.init_x - cell.x)**2 + 
                    (needle.init_y - cell.y)**2 + 
                    (needle.init_z - cell.z)**2
                )
        
        # 移动时间矩阵 t_cc' = d_cc' / v
        self.travel_time_matrix = self.dist_matrix / self.config.velocity
        
        # 针初始位置到细胞的移动时间 t_0c^p
        self.init_travel_time = self.init_dist / self.config.velocity
    
    def _precompute_indicators(self):
        """预计算碰撞和长距离指示器"""
        n = self.n_cells
        
        # 初始化碰撞类型距离矩阵
        self.dist_tip_tip = np.full((n, n), np.inf)      # 针尖-针尖距离
        self.dist_tip_body = np.full((n, n), np.inf)     # 针尖-针身距离
        self.dist_body_body = np.full((n, n), np.inf)    # 针身-针身距离
        
        # 碰撞风险指示器
        self.is_tip_tip_collision = np.zeros((n, n), dtype=bool)
        self.is_tip_body_collision = np.zeros((n, n), dtype=bool)
        self.is_body_body_collision = np.zeros((n, n), dtype=bool)
        self.is_collide = np.zeros((n, n), dtype=bool)   # 综合碰撞风险指示器 I^collide_cc'
        
        # 计算不同颜色细胞对的碰撞检测
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.cells[i].color == self.cells[j].color:
                    continue  # 同颜色由同一根针处理，不存在碰撞问题
                
                cell_i = self.cells[i]
                cell_j = self.cells[j]
                
                # 获取处理各细胞的针
                needle_i = self.needles[self.color_to_needle[cell_i.color]]
                needle_j = self.needles[self.color_to_needle[cell_j.color]]
                
                # 根据碰撞检测模式计算几何
                if self.config.collision_mode == CollisionDetectionMode.MODE_2D:
                    # 2D模式：忽略Z坐标和垂直角度
                    tip_i, base_i = self._get_needle_geometry_2d(needle_i, cell_i)
                    tip_j, base_j = self._get_needle_geometry_2d(needle_j, cell_j)
                else:
                    # 3D模式：完整三维几何
                    tip_i = needle_i.get_tip_position(cell_i)
                    base_i = needle_i.get_base_position(cell_i)
                    tip_j = needle_j.get_tip_position(cell_j)
                    base_j = needle_j.get_base_position(cell_j)
                
                # 计算三种碰撞类型的距离
                # 1. 针尖-针尖距离
                self.dist_tip_tip[i, j] = np.linalg.norm(tip_i - tip_j)
                
                # 2. 针尖-针身距离 (取两个方向的最小值)
                dist_tip_i_to_body_j, _ = point_to_segment_distance(tip_i, tip_j, base_j)
                dist_tip_j_to_body_i, _ = point_to_segment_distance(tip_j, tip_i, base_i)
                self.dist_tip_body[i, j] = min(dist_tip_i_to_body_j, dist_tip_j_to_body_i)
                
                # 3. 针身-针身距离
                self.dist_body_body[i, j], _, _ = segment_to_segment_distance(
                    tip_i, base_i, tip_j, base_j
                )
                
                # 判断碰撞风险
                self.is_tip_tip_collision[i, j] = self.dist_tip_tip[i, j] < self.config.d_tip
                self.is_tip_body_collision[i, j] = self.dist_tip_body[i, j] < self.config.d_tip
                self.is_body_body_collision[i, j] = self.dist_body_body[i, j] < self.config.d_body
                
                # 综合碰撞风险
                self.is_collide[i, j] = (
                    self.is_tip_tip_collision[i, j] or 
                    self.is_tip_body_collision[i, j] or 
                    self.is_body_body_collision[i, j]
                )
        
        # 保持向后兼容: is_close 作为 is_collide 的别名
        self.is_close = self.is_collide
        
        # I^long_cc': 长距离移动指示器
        self.is_long = self.dist_matrix > self.config.d_long
    
    def _get_needle_geometry_2d(self, needle: Needle, cell: Cell) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取2D模式下的针几何 (忽略Z坐标和垂直角度)
        
        Returns:
            tip: 针尖位置 (z=0)
            base: 针座位置 (z=0)
        """
        # 针尖位置 (z置零)
        tip = np.array([cell.x, cell.y, 0.0])
        
        # 2D模式下垂直角度为0，方向向量简化为
        # d_p = (cos(θ_xy), sin(θ_xy), 0)
        direction = np.array([
            np.cos(needle.theta_xy),
            np.sin(needle.theta_xy),
            0.0
        ])
        
        base = tip + needle.length * direction
        return tip, base
    
    def get_distance(self, cell1_id: int, cell2_id: int) -> float:
        """获取两细胞间距离 d_cc'"""
        return self.dist_matrix[cell1_id, cell2_id]
    
    def get_init_distance(self, needle_id: int, cell_id: int) -> float:
        """获取针初始位置到细胞的距离 d_0c^p"""
        return self.init_dist[needle_id, cell_id]
    
    def get_travel_time(self, cell1_id: int, cell2_id: int) -> float:
        """获取针在两细胞间移动的时间 t_cc'"""
        return self.travel_time_matrix[cell1_id, cell2_id]
    
    def get_init_travel_time(self, needle_id: int, cell_id: int) -> float:
        """获取针从初始位置移动到细胞的时间 t_0c^p"""
        return self.init_travel_time[needle_id, cell_id]
    
    def get_needle_for_color(self, color: int) -> int:
        """获取处理指定颜色的针 p(m)"""
        return self.color_to_needle[color]
    
    def get_cells_for_color(self, color: int) -> List[int]:
        """获取指定颜色的所有细胞 C_m"""
        return self.cells_by_color.get(color, [])
    
    def check_collision(self, cell1_id: int, cell2_id: int) -> Tuple[bool, str]:
        """
        检查两个细胞对应的针是否存在碰撞风险
        
        Args:
            cell1_id: 细胞1的ID
            cell2_id: 细胞2的ID
            
        Returns:
            (has_collision, collision_type): 
                - has_collision: 是否存在碰撞风险
                - collision_type: 碰撞类型 ("None", "TipToTip", "TipToBody", "BodyToBody")
        """
        if self.cells[cell1_id].color == self.cells[cell2_id].color:
            return False, "None"  # 同颜色由同一针处理，不存在碰撞
        
        if not self.is_collide[cell1_id, cell2_id]:
            return False, "None"
        
        # 按优先级判断碰撞类型 (与C++代码一致)
        if self.is_tip_tip_collision[cell1_id, cell2_id]:
            return True, "TipToTip"
        elif self.is_tip_body_collision[cell1_id, cell2_id]:
            return True, "TipToBody"
        elif self.is_body_body_collision[cell1_id, cell2_id]:
            return True, "BodyToBody"
        
        return False, "None"
    
    def get_collision_distances(self, cell1_id: int, cell2_id: int) -> Dict[str, float]:
        """
        获取两个细胞对应的针之间的各类碰撞距离
        
        Returns:
            包含三种距离的字典:
            - "tip_tip": 针尖-针尖距离
            - "tip_body": 针尖-针身距离
            - "body_body": 针身-针身距离
        """
        return {
            "tip_tip": self.dist_tip_tip[cell1_id, cell2_id],
            "tip_body": self.dist_tip_body[cell1_id, cell2_id],
            "body_body": self.dist_body_body[cell1_id, cell2_id]
        }


# =============================================================================
# 调度解码器
# =============================================================================

class ScheduleDecoder:
    """
    调度解码器
    
    将染色体 (优先级编码) 解码为可行调度方案
    处理所有约束:
    - 针-颜色绑定
    - 相机互斥 (Find阶段)
    - 碰撞避免
    - 并行执行 (Wait阶段)
    """
    
    def __init__(self, problem: CellInjectionProblem):
        self.problem = problem
        self.config = problem.config
    
    def decode(self, chromosome: np.ndarray) -> Tuple[List[CellTask], Dict]:
        """
        解码染色体为可行调度
        
        编码方案: 优先级编码
        - chromosome[i] ∈ [0,1] 表示细胞i的优先级
        - 同颜色细胞按优先级排序确定处理顺序 (TSP子序列)
        
        Args:
            chromosome: 长度为n_cells的优先级数组
            
        Returns:
            tasks: 任务调度列表
            metrics: 调度指标字典
        """
        # 1. 根据优先级确定每种颜色内的细胞处理顺序
        color_sequences = self._decode_sequences(chromosome)
        
        # 2. 使用事件驱动调度算法生成可行调度
        tasks = self._event_driven_schedule(color_sequences)
        
        # 3. 计算目标函数值
        metrics = self._compute_objectives(tasks, color_sequences)
        
        return tasks, metrics
    
    def _decode_sequences(self, chromosome: np.ndarray) -> Dict[int, List[int]]:
        """
        解码每种颜色的细胞处理顺序
        
        对于每种颜色m, 其细胞集C_m按优先级值排序
        形成该颜色的TSP子序列
        """
        color_sequences = {}
        
        for color, cell_ids in self.problem.cells_by_color.items():
            if not cell_ids:
                continue
            # 按优先级排序 (优先级值小的先处理)
            priorities = [(chromosome[cid], cid) for cid in cell_ids]
            priorities.sort()
            color_sequences[color] = [cid for _, cid in priorities]
        
        return color_sequences
    
    def _event_driven_schedule(self, color_sequences: Dict[int, List[int]]) -> List[CellTask]:
        """
        事件驱动的调度算法
        
        实现约束:
        - 相机互斥: 任意时刻最多一根针处于Find阶段
        - 碰撞避免: 距离过近的不同针任务时间窗口不能重叠
        - 序列约束: 同颜色细胞按指定顺序处理
        - 并行机制: 一针进入Wait阶段后其他针可获取相机
        """
        problem = self.problem
        config = self.config
        
        # 初始化针状态
        needle_state = {}
        for needle in problem.needles:
            color = needle.color
            sequence = color_sequences.get(color, [])
            needle_state[needle.id] = {
                'color': color,
                'queue': list(sequence),       # 待处理细胞队列
                'available_time': 0.0,         # 针可用时间 (上一任务完成后)
                'last_cell': None,             # 上一个处理的细胞 (用于计算移动时间)
                'active_task': None            # 当前活动任务 (处于Wait阶段)
            }
        
        # 全局状态
        camera_free_time = 0.0          # 相机可用时间
        active_tasks: List[CellTask] = []  # 当前活动的任务 (处于Wait阶段)
        completed_tasks: List[CellTask] = []  # 已完成的任务
        
        # 主调度循环
        max_iterations = self.problem.n_cells * 100  # 防止无限循环
        iteration = 0
        current_time = 0.0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查是否所有任务已完成
            all_done = all(
                len(ns['queue']) == 0 and ns['active_task'] is None 
                for ns in needle_state.values()
            )
            if all_done:
                break
            
            # 更新活动任务状态: 将已完成的任务移到completed
            new_active = []
            for task in active_tasks:
                if task.end_time <= current_time:
                    completed_tasks.append(task)
                    needle_state[task.needle_id]['active_task'] = None
                else:
                    new_active.append(task)
            active_tasks = new_active
            
            # 如果相机可用, 尝试调度新任务
            if camera_free_time <= current_time:
                new_task = self._schedule_next_task(
                    needle_state, active_tasks, current_time
                )
                
                if new_task is not None:
                    # 更新状态
                    needle_state[new_task.needle_id]['queue'].pop(0)
                    needle_state[new_task.needle_id]['last_cell'] = new_task.cell_id
                    needle_state[new_task.needle_id]['available_time'] = new_task.end_time
                    needle_state[new_task.needle_id]['active_task'] = new_task
                    
                    active_tasks.append(new_task)
                    camera_free_time = new_task.find_end_time  # 相机在Find阶段结束后释放
                    continue  # 继续尝试调度（可能还有其他可调度任务，但需要等待相机释放）
            
            # 推进时间到下一个事件点
            next_time = self._find_next_schedule_time(
                needle_state, active_tasks, camera_free_time, current_time + 0.001
            )
            if next_time <= current_time:
                # 无法推进，退出
                break
            current_time = next_time
        
        # 收集所有剩余的活动任务
        completed_tasks.extend(active_tasks)
        
        return completed_tasks
    
    def _find_next_schedule_time(
        self, 
        needle_state: Dict, 
        active_tasks: List[CellTask],
        camera_free_time: float,
        min_time: float = 0.0
    ) -> float:
        """
        找到下一个可调度的时间点 (>= min_time)
        
        候选时间点包括:
        1. 相机释放时间 (camera_free_time)
        2. 活动任务的Find结束时间 (相机释放)
        3. 活动任务的完成时间 (针释放)
        """
        candidate_times = set()
        candidate_times.add(camera_free_time)
        
        # 活动任务的结束时间和Find结束时间
        for task in active_tasks:
            candidate_times.add(task.end_time)
            candidate_times.add(task.find_end_time)
        
        # 只考虑 >= min_time 的时间点
        valid_times = [t for t in candidate_times if t >= min_time]
        return min(valid_times) if valid_times else min_time
    
    def _schedule_next_task(
        self,
        needle_state: Dict,
        active_tasks: List[CellTask],
        current_time: float
    ) -> Optional[CellTask]:
        """
        选择并创建下一个要执行的任务
        
        贪心策略: 选择最早可开始的任务
        需要检查碰撞约束
        
        关键约束:
        - 相机互斥: Find阶段需要独占相机
        - 并行Wait: 多针可同时处于Wait阶段
        - 同一针: 必须等当前任务完成才能开始下一个
        """
        problem = self.problem
        config = self.config
        
        best_task = None
        best_start_time = float('inf')
        
        for needle_id, ns in needle_state.items():
            if not ns['queue']:
                continue  # 该针无待处理任务
            
            # 检查该针是否有正在执行的任务
            if ns['active_task'] is not None:
                continue  # 该针正在处理任务,必须等待完成
            
            next_cell = ns['queue'][0]
            
            # 计算移动时间
            if ns['last_cell'] is None:
                travel_time = problem.get_init_travel_time(needle_id, next_cell)
            else:
                travel_time = problem.get_travel_time(ns['last_cell'], next_cell)
            
            # 计算最早开始时间 (考虑针可用时间和移动时间)
            # available_time 是上一个任务的结束时间
            earliest_start = max(current_time, ns['available_time'] + travel_time)
            
            # 检查碰撞约束: γ_cc' = 0 if I^close_cc' = 1
            collision_free_start = self._find_collision_free_start(
                next_cell, earliest_start, active_tasks, needle_id
            )
            
            if collision_free_start < best_start_time:
                best_start_time = collision_free_start
                best_task = CellTask(
                    cell_id=next_cell,
                    needle_id=needle_id,
                    color=problem.cells[next_cell].color,
                    start_time=collision_free_start,
                    find_end_time=collision_free_start + config.t_find,
                    end_time=collision_free_start + config.t_find + config.t_wait
                )
        
        return best_task
    
    def _find_collision_free_start(
        self,
        cell_id: int,
        earliest_start: float,
        active_tasks: List[CellTask],
        needle_id: int
    ) -> float:
        """
        找到满足碰撞约束的最早开始时间
        
        约束: 若 I^collide_cc' = 1 (存在针尖-针尖/针尖-针身/针身-针身碰撞风险), 
              则 γ_cc' = 0 (时间窗口不能重叠)
        """
        problem = self.problem
        config = self.config
        
        start_time = earliest_start
        task_duration = config.t_find + config.t_wait
        
        # 检查与所有活动任务的碰撞
        for active_task in active_tasks:
            # 只检查不同针 (不同颜色) 的任务
            if active_task.needle_id == needle_id:
                continue
            
            # 检查是否存在碰撞风险 (综合三种碰撞类型)
            if problem.is_collide[cell_id, active_task.cell_id]:
                # 需要确保时间窗口不重叠
                # 新任务窗口: [start_time, start_time + duration]
                # 活动任务窗口: [active_task.start_time, active_task.end_time]
                
                # 如果有重叠, 推迟开始时间到活动任务结束
                if start_time < active_task.end_time and \
                   start_time + task_duration > active_task.start_time:
                    start_time = max(start_time, active_task.end_time)
        
        return start_time
    
    def _compute_objectives(
        self, 
        tasks: List[CellTask], 
        color_sequences: Dict[int, List[int]]
    ) -> Dict:
        """
        计算四个目标函数值
        
        f1: C_max - 最大完成时间 (makespan)
        f2: D_total - 总移动距离
        f3: N_long - 长距离移动次数
        f4: T_idle_total - 总空闲时间
        """
        problem = self.problem
        config = self.config
        
        if not tasks:
            return {
                'makespan': float('inf'),
                'total_distance': float('inf'),
                'long_moves': float('inf'),
                'total_idle': float('inf')
            }
        
        # f1: 最大完成时间 C_max = max_c {e_c}
        makespan = max(t.end_time for t in tasks)
        
        # f2: 总移动距离
        total_distance = 0.0
        for color, sequence in color_sequences.items():
            if not sequence:
                continue
            needle_id = problem.get_needle_for_color(color)
            
            # 从初始位置到第一个细胞
            total_distance += problem.get_init_distance(needle_id, sequence[0])
            
            # 序列内细胞间移动
            for i in range(len(sequence) - 1):
                total_distance += problem.get_distance(sequence[i], sequence[i + 1])
        
        # f3: 长距离移动次数
        long_moves = 0
        for color, sequence in color_sequences.items():
            for i in range(len(sequence) - 1):
                if problem.is_long[sequence[i], sequence[i + 1]]:
                    long_moves += 1
        
        # f4: 总空闲时间
        total_idle = 0.0
        for color, sequence in color_sequences.items():
            if not sequence:
                continue
            
            # 找到该颜色任务的时间范围
            color_tasks = [t for t in tasks if t.color == color]
            if color_tasks:
                S_m = min(t.start_time for t in color_tasks)  # min s_c
                E_m = max(t.end_time for t in color_tasks)    # max e_c
                T_work = len(sequence) * (config.t_find + config.t_wait)
                T_span = E_m - S_m
                total_idle += max(0, T_span - T_work)
        
        return {
            'makespan': makespan,
            'total_distance': total_distance,
            'long_moves': float(long_moves),
            'total_idle': total_idle
        }


# =============================================================================
# pymoo Problem定义
# =============================================================================

class CellInjectionOptProblem(Problem):
    """
    pymoo多目标优化问题定义
    
    决策变量: n_cells个实数 ∈ [0,1], 表示各细胞的调度优先级
    目标函数: 4个 (全部最小化)
    约束: 通过解码器隐式满足
    """
    
    def __init__(self, problem: CellInjectionProblem):
        self.injection_problem = problem
        self.decoder = ScheduleDecoder(problem)
        
        n_var = problem.n_cells
        n_obj = 4  # makespan, distance, long_moves, idle_time
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=0,  # 约束通过解码器满足
            xl=0.0,
            xu=1.0,
            vtype=float
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """批量评估解"""
        n_solutions = X.shape[0]
        F = np.zeros((n_solutions, self.n_obj))
        
        for i in range(n_solutions):
            _, metrics = self.decoder.decode(X[i])
            F[i, 0] = metrics['makespan']
            F[i, 1] = metrics['total_distance']
            F[i, 2] = metrics['long_moves']
            F[i, 3] = metrics['total_idle']
        
        out["F"] = F


# =============================================================================
# 自定义遗传算子
# =============================================================================

class PrioritySampling(Sampling):
    """优先级编码的随机采样"""
    
    def _do(self, problem, n_samples, **kwargs):
        return np.random.random((n_samples, problem.n_var))


class PriorityCrossover(Crossover):
    """
    优先级编码的交叉算子
    使用模拟二进制交叉 (SBX) 变体
    """
    
    def __init__(self, prob: float = 0.9, eta: float = 15):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob
        self.eta = eta
    
    def _do(self, problem, X, **kwargs):
        # X shape is (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        Y = np.zeros((self.n_offsprings, n_matings, n_var))
        
        for i in range(n_matings):
            p1, p2 = X[0, i].copy(), X[1, i].copy()
            c1, c2 = p1.copy(), p2.copy()
            
            if np.random.random() < self.prob:
                # 对每个基因位独立进行SBX交叉
                for j in range(n_var):
                    if np.random.random() < 0.5:
                        if abs(p1[j] - p2[j]) > 1e-14:
                            # SBX交叉
                            if p1[j] < p2[j]:
                                y1, y2 = p1[j], p2[j]
                            else:
                                y1, y2 = p2[j], p1[j]
                            
                            beta = 1.0 + (2.0 * y1 / (y2 - y1 + 1e-14))
                            alpha = 2.0 - beta ** (-(self.eta + 1))
                            
                            u = np.random.random()
                            if u <= (1.0 / alpha):
                                betaq = (u * alpha) ** (1.0 / (self.eta + 1))
                            else:
                                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta + 1))
                            
                            c1[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                            c2[j] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                            
                            c1[j] = np.clip(c1[j], 0, 1)
                            c2[j] = np.clip(c2[j], 0, 1)
            
            Y[0, i] = c1
            Y[1, i] = c2
        
        return Y


class PriorityMutation(Mutation):
    """
    优先级编码的变异算子
    使用多项式变异
    """
    
    def __init__(self, prob: float = None, eta: float = 20):
        super().__init__()
        self.prob = prob
        self.eta = eta
    
    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        n_var = problem.n_var
        prob = self.prob if self.prob is not None else 1.0 / n_var
        
        for i in range(len(X)):
            for j in range(n_var):
                if np.random.random() < prob:
                    y = Y[i, j]
                    delta1 = y
                    delta2 = 1.0 - y
                    
                    u = np.random.random()
                    if u < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (self.eta + 1))
                        deltaq = val ** (1.0 / (self.eta + 1)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (self.eta + 1))
                        deltaq = 1.0 - val ** (1.0 / (self.eta + 1))
                    
                    Y[i, j] = np.clip(y + deltaq, 0, 1)
        
        return Y


class ColorAwareCrossover(Crossover):
    """
    颜色感知交叉算子
    保持同颜色细胞的相对顺序关系
    """
    
    def __init__(self, prob: float = 0.9):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        # X shape is (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        Y = np.zeros((self.n_offsprings, n_matings, n_var))
        injection_problem = problem.injection_problem
        
        for i in range(n_matings):
            p1, p2 = X[0, i].copy(), X[1, i].copy()
            c1, c2 = p1.copy(), p2.copy()
            
            if np.random.random() < self.prob:
                # 对每种颜色独立进行顺序交叉
                for color, cell_ids in injection_problem.cells_by_color.items():
                    if len(cell_ids) <= 1:
                        continue
                    
                    # 获取两个父代中该颜色细胞的顺序
                    order1 = sorted(cell_ids, key=lambda x: p1[x])
                    order2 = sorted(cell_ids, key=lambda x: p2[x])
                    
                    if np.random.random() < 0.5:
                        # 部分映射交叉 (PMX) 思想
                        n = len(cell_ids)
                        pt1, pt2 = sorted(np.random.choice(n, 2, replace=False))
                        
                        # 子代1: 从p1继承中间段顺序关系
                        # 子代2: 从p2继承中间段顺序关系
                        new_order1 = order2[:pt1] + order1[pt1:pt2] + order2[pt2:]
                        new_order2 = order1[:pt1] + order2[pt1:pt2] + order1[pt2:]
                        
                        # 修复重复
                        new_order1 = self._repair_order(new_order1, cell_ids)
                        new_order2 = self._repair_order(new_order2, cell_ids)
                        
                        # 转换回优先级值
                        for idx, cid in enumerate(new_order1):
                            c1[cid] = (idx + 0.5) / n
                        for idx, cid in enumerate(new_order2):
                            c2[cid] = (idx + 0.5) / n
            
            Y[0, i] = c1
            Y[1, i] = c2
        
        return Y
    
    def _repair_order(self, order: List[int], valid_ids: List[int]) -> List[int]:
        """修复顺序中的重复元素"""
        seen = set()
        result = []
        missing = set(valid_ids)
        
        for item in order:
            if item in valid_ids and item not in seen:
                result.append(item)
                seen.add(item)
                missing.discard(item)
        
        result.extend(missing)
        return result


class ColorAwareMutation(Mutation):
    """
    颜色感知变异算子
    在同颜色细胞内进行2-opt变异
    """
    
    def __init__(self, prob: float = 0.3):
        super().__init__()
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        injection_problem = problem.injection_problem
        
        for i in range(len(X)):
            for color, cell_ids in injection_problem.cells_by_color.items():
                if len(cell_ids) <= 2:
                    continue
                
                if np.random.random() < self.prob:
                    # 获取当前顺序
                    order = sorted(cell_ids, key=lambda x: Y[i, x])
                    n = len(order)
                    
                    # 2-opt: 随机选择两个位置进行反转
                    pt1, pt2 = sorted(np.random.choice(n, 2, replace=False))
                    order[pt1:pt2+1] = order[pt1:pt2+1][::-1]
                    
                    # 转换回优先级值
                    for idx, cid in enumerate(order):
                        Y[i, cid] = (idx + 0.5) / n
        
        return Y


# =============================================================================
# 回调和工具类
# =============================================================================

class ProgressCallback(Callback):
    """进度回调"""
    
    def __init__(self, n_gen_total: int, print_freq: int = 10):
        super().__init__()
        self.n_gen_total = n_gen_total
        self.print_freq = print_freq
    
    def notify(self, algorithm):
        if algorithm.n_gen % self.print_freq == 0:
            F = algorithm.pop.get("F")
            print(f"Generation {algorithm.n_gen:4d}/{self.n_gen_total}: "
                  f"Pop size = {len(F)}, "
                  f"Makespan = [{F[:, 0].min():.1f}, {F[:, 0].max():.1f}], "
                  f"Distance = [{F[:, 1].min():.1f}, {F[:, 1].max():.1f}]")


# =============================================================================
# 主求解函数
# =============================================================================

def solve_cell_injection(
    problem: CellInjectionProblem,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = None,
    verbose: bool = True,
    use_color_aware_operators: bool = True
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    使用NSGA-II求解多针细胞灌注调度问题
    
    Args:
        problem: 问题实例
        pop_size: 种群大小
        n_gen: 最大迭代数
        seed: 随机种子
        verbose: 是否打印进度
        use_color_aware_operators: 是否使用颜色感知的遗传算子
        
    Returns:
        X: Pareto最优解的决策变量
        F: 对应的目标函数值
        schedules: 对应的调度方案列表
    """
    # 创建pymoo问题
    opt_problem = CellInjectionOptProblem(problem)
    
    # 选择遗传算子
    if use_color_aware_operators:
        crossover = ColorAwareCrossover(prob=0.9)
        mutation = ColorAwareMutation(prob=0.3)
    else:
        crossover = PriorityCrossover(prob=0.9, eta=15)
        mutation = PriorityMutation(eta=20)
    
    # 配置NSGA-II
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PrioritySampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )
    
    # 回调
    callback = ProgressCallback(n_gen, print_freq=20) if verbose else None
    
    # 执行优化
    if verbose:
        print(f"开始NSGA-II优化: 种群={pop_size}, 代数={n_gen}")
    
    res = minimize(
        opt_problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # 解码所有Pareto最优解
    decoder = ScheduleDecoder(problem)
    schedules = []
    for x in res.X:
        tasks, _ = decoder.decode(x)
        schedules.append(tasks)
    
    return res.X, res.F, schedules


# =============================================================================
# 测试数据生成
# =============================================================================

def create_test_instance(
    n_cells: int = 50,
    n_colors: int = 4,
    field_size: float = 1000.0,
    seed: int = 42
) -> CellInjectionProblem:
    """
    创建测试问题实例
    
    Args:
        n_cells: 细胞总数
        n_colors: 颜色数量 (= 针数量)
        field_size: 视野大小 (μm)
        seed: 随机种子
        
    Returns:
        CellInjectionProblem实例
    """
    np.random.seed(seed)
    
    # 生成细胞 - 使用图着色分配颜色
    cells = []
    for i in range(n_cells):
        cell = Cell(
            id=i,
            x=np.random.uniform(0, field_size),
            y=np.random.uniform(0, field_size),
            z=np.random.uniform(0, 50),  # 组织厚度约50μm
            color=i % n_colors  # 简单的循环分配 (实际应用中由图着色算法决定)
        )
        cells.append(cell)
    
    # 生成针 - 均匀分布在镜头周围
    needles = []
    for i in range(n_colors):
        angle = 2 * np.pi * i / n_colors
        needle = Needle(
            id=i,
            color=i,
            init_x=field_size / 2 + 300 * np.cos(angle),  # 待机位置
            init_y=field_size / 2 + 300 * np.sin(angle),
            init_z=200,  # 待机高度
            theta_xy=angle,
            theta_z=np.pi / 4,  # 45度倾斜
            length=200000
        )
        needles.append(needle)
    
    config = ProblemConfig(
        t_find=60.0,        # 1分钟
        t_wait=240.0,       # 4分钟
        velocity=10.0,      # 10 μm/s
        d_tip=50.0,         # 针尖安全距离 (用于针尖-针尖、针尖-针身碰撞检测)
        d_body=100.0,       # 针身安全距离 (用于针身-针身碰撞检测)
        d_long=500.0,       # 长距离阈值
        collision_mode=CollisionDetectionMode.MODE_2D  # 碰撞检测模式
    )
    
    return CellInjectionProblem(cells, needles, config)


# =============================================================================
# 结果分析和可视化
# =============================================================================

def analyze_results(
    F: np.ndarray,
    schedules: List[List[CellTask]],
    problem: CellInjectionProblem
) -> Dict:
    """
    分析优化结果
    
    Returns:
        分析结果字典
    """
    n_solutions = len(F)
    
    # 目标函数统计
    analysis = {
        'n_pareto_solutions': n_solutions,
        'makespan': {
            'min': F[:, 0].min(),
            'max': F[:, 0].max(),
            'mean': F[:, 0].mean(),
            'best_idx': np.argmin(F[:, 0])
        },
        'total_distance': {
            'min': F[:, 1].min(),
            'max': F[:, 1].max(),
            'mean': F[:, 1].mean(),
            'best_idx': np.argmin(F[:, 1])
        },
        'long_moves': {
            'min': F[:, 2].min(),
            'max': F[:, 2].max(),
            'mean': F[:, 2].mean(),
            'best_idx': np.argmin(F[:, 2])
        },
        'total_idle': {
            'min': F[:, 3].min(),
            'max': F[:, 3].max(),
            'mean': F[:, 3].mean(),
            'best_idx': np.argmin(F[:, 3])
        }
    }
    
    return analysis


def print_schedule(tasks: List[CellTask], problem: CellInjectionProblem):
    """打印调度方案"""
    print("\n" + "=" * 70)
    print("调度方案详情")
    print("=" * 70)
    
    # 按针分组
    by_needle = {}
    for task in sorted(tasks, key=lambda t: t.start_time):
        if task.needle_id not in by_needle:
            by_needle[task.needle_id] = []
        by_needle[task.needle_id].append(task)
    
    for needle_id in sorted(by_needle.keys()):
        needle_tasks = by_needle[needle_id]
        color = problem.needle_to_color[needle_id]
        print(f"\n针 {needle_id} (颜色 {color}):")
        print("-" * 60)
        print(f"{'细胞ID':>8} {'开始时间':>12} {'Find结束':>12} {'完成时间':>12}")
        print("-" * 60)
        for task in needle_tasks:
            print(f"{task.cell_id:>8} {task.start_time:>12.1f} "
                  f"{task.find_end_time:>12.1f} {task.end_time:>12.1f}")
    
    # 时间线摘要
    if tasks:
        makespan = max(t.end_time for t in tasks)
        print(f"\n总完成时间 (Makespan): {makespan:.1f} 秒 ({makespan/60:.1f} 分钟)")


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """主函数 - 演示完整求解流程"""
    
    print("=" * 70)
    print("多针细胞染料灌注调度问题 - NSGA-II求解器")
    print("=" * 70)
    
    # 1. 创建问题实例
    print("\n[1] 创建问题实例...")
    problem = create_test_instance(
        n_cells=40,      # 40个细胞
        n_colors=4,      # 4种颜色/4根针
        field_size=1000, # 1000μm视野
        seed=42
    )
    
    print(f"    细胞总数: {problem.n_cells}")
    print(f"    颜色/针数量: {problem.n_colors}")
    for color, cells in problem.cells_by_color.items():
        print(f"    颜色 {color}: {len(cells)} 个细胞")
    
    # 2. 执行NSGA-II优化
    print("\n[2] 执行NSGA-II多目标优化...")
    X, F, schedules = solve_cell_injection(
        problem,
        pop_size=100,
        n_gen=150,
        seed=42,
        verbose=True,
        use_color_aware_operators=True
    )
    
    # 3. 分析结果
    print("\n[3] 结果分析")
    print("=" * 70)
    analysis = analyze_results(F, schedules, problem)
    
    print(f"\nPareto最优解数量: {analysis['n_pareto_solutions']}")
    
    print("\n目标函数范围:")
    print(f"  f1 (Makespan):     [{analysis['makespan']['min']:.1f}, "
          f"{analysis['makespan']['max']:.1f}] 秒")
    print(f"  f2 (总移动距离):    [{analysis['total_distance']['min']:.1f}, "
          f"{analysis['total_distance']['max']:.1f}] μm")
    print(f"  f3 (长距离移动):    [{analysis['long_moves']['min']:.0f}, "
          f"{analysis['long_moves']['max']:.0f}] 次")
    print(f"  f4 (总空闲时间):    [{analysis['total_idle']['min']:.1f}, "
          f"{analysis['total_idle']['max']:.1f}] 秒")
    
    # 4. 打印代表性解
    print("\n" + "=" * 70)
    print("代表性Pareto最优解")
    print("=" * 70)
    
    # 最小makespan的解
    idx = analysis['makespan']['best_idx']
    print(f"\n最小Makespan解 (索引 {idx}):")
    print(f"  Makespan:     {F[idx, 0]:.1f} 秒 ({F[idx, 0]/60:.1f} 分钟)")
    print(f"  总移动距离:   {F[idx, 1]:.1f} μm")
    print(f"  长距离移动:   {F[idx, 2]:.0f} 次")
    print(f"  总空闲时间:   {F[idx, 3]:.1f} 秒")
    
    # 最小距离的解
    idx = analysis['total_distance']['best_idx']
    print(f"\n最小总移动距离解 (索引 {idx}):")
    print(f"  Makespan:     {F[idx, 0]:.1f} 秒 ({F[idx, 0]/60:.1f} 分钟)")
    print(f"  总移动距离:   {F[idx, 1]:.1f} μm")
    print(f"  长距离移动:   {F[idx, 2]:.0f} 次")
    print(f"  总空闲时间:   {F[idx, 3]:.1f} 秒")
    
    # 打印最小makespan解的详细调度
    best_makespan_idx = analysis['makespan']['best_idx']
    print_schedule(schedules[best_makespan_idx], problem)
    
    print("\n" + "=" * 70)
    print("优化完成!")
    print("=" * 70)
    
    return X, F, schedules, problem


if __name__ == "__main__":
    X, F, schedules, problem = main()
