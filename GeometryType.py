from enum import Enum
import math
from typing import List, Tuple


class Point2D:
    """2D点类"""
    
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Point2D(self.x / scalar, self.y / scalar)
    
    def __neg__(self):
        return Point2D(-self.x, -self.y)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"Point2D({self.x}, {self.y})"
    
    def dot(self, other):
        """点积"""
        return self.x * other.x + self.y * other.y
    
    def length(self):
        """向量长度"""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        """归一化"""
        len_val = self.length()
        if len_val == 0:
            return Point2D(0, 0)
        return Point2D(self.x / len_val, self.y / len_val)
    
    def distance(self, other):
        """到另一个点的距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    @staticmethod
    def lerp(a, b, t):
        """线性插值"""
        return a * (1 - t) + b * t
    
    def reflect(self, normal):
        """反射"""
        return self - normal * (2 * self.dot(normal))


class Point3D:
    """3D点类"""
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self):
        return Point3D(-self.x, -self.y, -self.z)
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __repr__(self):
        return f"Point3D({self.x}, {self.y}, {self.z})"
    
    def dot(self, other):
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """叉积"""
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        """向量长度"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self):
        """归一化"""
        len_val = self.length()
        if len_val == 0:
            return Point3D(0, 0, 0)
        return Point3D(self.x / len_val, self.y / len_val, self.z / len_val)
    
    def distance(self, other):
        """到另一个点的距离"""
        return (self - other).length()
    
    @staticmethod
    def lerp(a, b, t):
        """线性插值"""
        return a * (1 - t) + b * t
    
    def reflect(self, normal):
        """反射"""
        return self - normal * (2 * self.dot(normal))
    
    def approx_equal(self, other, epsilon=1e-6):
        """近似相等比较"""
        return (abs(self.x - other.x) < epsilon and
                abs(self.y - other.y) < epsilon and
                abs(self.z - other.z) < epsilon)


# 类型别名
Point3Df = Point3D


class Polygon:
    """多边形类"""
    
    def __init__(self, points: List[Point2D] = None):
        self.vertices = points if points is not None else []
        self.area = 0.0
        self.centroid = Point2D(0, 0)
        if self.vertices:
            self._calculate_properties()
    
    def intersects(self, other: 'Polygon') -> bool:
        """检查当前多边形是否与另一个多边形相交"""
        # 检查边是否相交
        for i in range(len(self.vertices)):
            for j in range(len(other.vertices)):
                if self._do_intersect(
                    self.vertices[i], self.vertices[(i + 1) % len(self.vertices)],
                    other.vertices[j], other.vertices[(j + 1) % len(other.vertices)]
                ):
                    return True
        
        # 检查一个多边形是否完全在另一个内部
        if self._is_inside(other.vertices[0]) or other._is_inside(self.vertices[0]):
            return True
        
        return False
    
    def add_vertex(self, point: Point2D):
        """向多边形添加新的顶点"""
        self.vertices.append(point)
        self._calculate_properties()
    
    def get_vertices(self) -> List[Point2D]:
        """获取多边形的所有顶点"""
        return self.vertices
    
    def get_area(self) -> float:
        """获取多边形的面积"""
        return self.area
    
    def get_centroid(self) -> Point2D:
        """获取多边形的质心"""
        return self.centroid
    
    def contains_point(self, point: Point2D) -> bool:
        """检查指定点是否在多边形内部"""
        return self._is_inside(point)
    
    def get_bounding_box_aabb(self) -> Tuple[Point2D, Point2D]:
        """获取多边形的轴对齐包围盒"""
        if not self.vertices:
            return Point2D(0, 0), Point2D(0, 0)
        
        min_point = Point2D(self.vertices[0].x, self.vertices[0].y)
        max_point = Point2D(self.vertices[0].x, self.vertices[0].y)
        
        for vertex in self.vertices:
            min_point.x = min(min_point.x, vertex.x)
            min_point.y = min(min_point.y, vertex.y)
            max_point.x = max(max_point.x, vertex.x)
            max_point.y = max(max_point.y, vertex.y)
        
        return min_point, max_point
    
    def get_perimeter(self) -> float:
        """计算多边形的周长"""
        perimeter = 0.0
        n = len(self.vertices)
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            perimeter += math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        return perimeter
    
    def is_convex(self) -> bool:
        """检查多边形是否为凸多边形"""
        n = len(self.vertices)
        if n < 3:
            return False
        
        sign = None
        
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            p3 = self.vertices[(i + 2) % n]
            
            cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
            
            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False
        
        return True
    
    def _calculate_properties(self):
        """计算多边形的基本属性（面积和质心）"""
        self.area = 0.0
        cx = 0.0
        cy = 0.0
        n = len(self.vertices)
        
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            
            cross = p1.x * p2.y - p2.x * p1.y
            self.area += cross
            
            cx += (p1.x + p2.x) * cross
            cy += (p1.y + p2.y) * cross
        
        self.area = abs(self.area) / 2
        
        if abs(self.area) > 1e-10:
            signed_area = self.area if self.area > 0 else -self.area
            self.centroid.x = cx / (6 * signed_area)
            self.centroid.y = cy / (6 * signed_area)
    
    @staticmethod
    def _orientation(p: Point2D, q: Point2D, r: Point2D) -> int:
        """计算三点的方向关系"""
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  # 共线
        return 1 if val > 0 else 2  # 顺时针或逆时针
    
    @staticmethod
    def _on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
        """检查点q是否在线段pr上"""
        if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
            return True
        return False
    
    def _do_intersect(self, p1: Point2D, q1: Point2D, p2: Point2D, q2: Point2D) -> bool:
        """检查两条线段是否相交"""
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)
        
        if o1 != o2 and o3 != o4:
            return True
        
        if o1 == 0 and self._on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self._on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self._on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self._on_segment(p2, q1, q2):
            return True
        
        return False
    
    def _is_inside(self, p: Point2D) -> bool:
        """使用射线法检查点是否在多边形内部"""
        n = len(self.vertices)
        if n < 3:
            return False
        
        extreme = Point2D(1e9, p.y)
        count = 0
        i = 0
        
        while True:
            next_i = (i + 1) % n
            
            if self._do_intersect(self.vertices[i], self.vertices[next_i], p, extreme):
                if self._orientation(self.vertices[i], p, self.vertices[next_i]) == 0:
                    return self._on_segment(self.vertices[i], p, self.vertices[next_i])
                count += 1
            
            i = next_i
            if i == 0:
                break
        
        return count % 2 == 1


class GeometryUtils:
    """几何工具函数类"""
    
    @staticmethod
    def closest_point_on_segment(point: Point3Df, seg_start: Point3Df, seg_end: Point3Df) -> Point3Df:
        """点到线段的最近点"""
        seg = seg_end - seg_start
        seg_len_sq = seg.dot(seg)
        
        if seg_len_sq < 1e-10:
            return seg_start
        
        t = max(0.0, min(1.0, (point - seg_start).dot(seg) / seg_len_sq))
        return seg_start + seg * t
    
    @staticmethod
    def point_to_segment_distance(point: Point3Df, seg_start: Point3Df, seg_end: Point3Df) -> float:
        """点到线段的距离"""
        closest = GeometryUtils.closest_point_on_segment(point, seg_start, seg_end)
        return point.distance(closest)
    
    @staticmethod
    def segment_to_segment_distance(p1: Point3Df, q1: Point3Df, p2: Point3Df, q2: Point3Df) -> Tuple[float, float, float]:
        """两条线段之间的最近距离和最近点参数
        
        返回: (最近距离, 线段1上的参数t1, 线段2上的参数t2)
        """
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        
        a = d1.dot(d1)
        e = d2.dot(d2)
        f = d2.dot(r)
        
        if a < 1e-10 and e < 1e-10:
            return p1.distance(p2), 0.0, 0.0
        
        if a < 1e-10:
            s = 0.0
            t = max(0.0, min(1.0, f / e))
        else:
            c = d1.dot(r)
            if e < 1e-10:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            else:
                b = d1.dot(d2)
                denom = a * e - b * b
                
                if abs(denom) > 1e-10:
                    s = max(0.0, min(1.0, (b * f - c * e) / denom))
                else:
                    s = 0.0
                
                t = (b * s + f) / e
                
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
        
        closest1 = p1 + d1 * s
        closest2 = p2 + d2 * t
        distance = closest1.distance(closest2)
        
        return distance, s, t
    
    @staticmethod
    def capsule_capsule_collision(p1: Point3Df, q1: Point3Df, r1: float, 
                                  p2: Point3Df, q2: Point3Df, r2: float) -> bool:
        """胶囊体与胶囊体碰撞检测"""
        dist, _, _ = GeometryUtils.segment_to_segment_distance(p1, q1, p2, q2)
        return dist < (r1 + r2)
    
    @staticmethod
    def capsule_sphere_collision(cap_start: Point3Df, cap_end: Point3Df, cap_radius: float,
                                sphere_center: Point3Df, sphere_radius: float) -> bool:
        """胶囊体与球体碰撞检测"""
        dist = GeometryUtils.point_to_segment_distance(sphere_center, cap_start, cap_end)
        return dist < (cap_radius + sphere_radius)
    
    @staticmethod
    def segment_sphere_intersection(seg_start: Point3Df, seg_end: Point3Df, 
                                   sphere_center: Point3Df, sphere_radius: float) -> bool:
        """线段与球体相交检测"""
        d = seg_end - seg_start
        f = seg_start - sphere_center
        
        a = d.dot(d)
        b = 2.0 * f.dot(d)
        c = f.dot(f) - sphere_radius * sphere_radius
        
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return False
        
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)
    
    @staticmethod
    def segment_aabb_intersection(seg_start: Point3Df, seg_end: Point3Df, 
                                 box_min: Point3Df, box_max: Point3Df) -> bool:
        """线段与AABB盒相交检测"""
        d = seg_end - seg_start
        inv_d = Point3Df(
            1.0 / d.x if abs(d.x) > 1e-10 else (1e10 if d.x >= 0 else -1e10),
            1.0 / d.y if abs(d.y) > 1e-10 else (1e10 if d.y >= 0 else -1e10),
            1.0 / d.z if abs(d.z) > 1e-10 else (1e10 if d.z >= 0 else -1e10)
        )
        
        t1 = (box_min.x - seg_start.x) * inv_d.x
        t2 = (box_max.x - seg_start.x) * inv_d.x
        t3 = (box_min.y - seg_start.y) * inv_d.y
        t4 = (box_max.y - seg_start.y) * inv_d.y
        t5 = (box_min.z - seg_start.z) * inv_d.z
        t6 = (box_max.z - seg_start.z) * inv_d.z
        
        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
        
        if tmax < 0 or tmin > tmax:
            return False
        return tmin <= 1.0 and tmax >= 0.0
    
    @staticmethod
    def capsule_aabb_collision(cap_start: Point3Df, cap_end: Point3Df, cap_radius: float,
                              box_center: Point3Df, box_half_size: Point3Df) -> bool:
        """胶囊体与AABB盒碰撞检测"""
        expanded_min = box_center - box_half_size - Point3Df(cap_radius, cap_radius, cap_radius)
        expanded_max = box_center + box_half_size + Point3Df(cap_radius, cap_radius, cap_radius)
        
        if GeometryUtils.segment_aabb_intersection(cap_start, cap_end, expanded_min, expanded_max):
            closest_on_seg = GeometryUtils.closest_point_on_segment(box_center, cap_start, cap_end)
            
            closest_on_box = Point3Df(
                max(box_center.x - box_half_size.x, min(closest_on_seg.x, box_center.x + box_half_size.x)),
                max(box_center.y - box_half_size.y, min(closest_on_seg.y, box_center.y + box_half_size.y)),
                max(box_center.z - box_half_size.z, min(closest_on_seg.z, box_center.z + box_half_size.z))
            )
            
            dist = GeometryUtils.point_to_segment_distance(closest_on_box, cap_start, cap_end)
            return dist < cap_radius
        return False
    
    @staticmethod
    def calculate_needle_direction_3d(angle_h: float, angle_v: float) -> Point3Df:
        """计算针身方向向量（从针尖指向针身末端）
        
        Args:
            angle_h: 水平角度（度）
            angle_v: 垂直角度（度）
        """
        rad_h = math.radians(angle_h)
        rad_v = math.radians(angle_v)
        
        direction = Point3Df()
        direction.x = math.cos(rad_v) * math.cos(rad_h)
        direction.y = math.cos(rad_v) * math.sin(rad_h)
        direction.z = math.sin(rad_v)
        
        return direction
    
    @staticmethod
    def calculate_needle_direction_2d(angle_h: float) -> Point3Df:
        """计算针身方向向量（2D版本，仅考虑水平角度）
        
        Args:
            angle_h: 水平角度（度）
        """
        rad_h = math.radians(angle_h)
        return Point3Df(math.cos(rad_h), math.sin(rad_h), 0)


class CollisionDetectionMode(Enum):
    Mode2D = 1
    Mode3D = 2


class NeedleGeometry:
    """针的几何表示"""
    
    def __init__(self, tip: Point3Df = None, angle_h: float = 0.0, angle_v: float = 0.0,
                 body_length: float = 0.0, body_radius: float = 0.0,
                 collision_mode: CollisionDetectionMode = CollisionDetectionMode.Mode2D):
        """
        Args:
            tip: 针尖位置
            angle_h: 水平角度（度）- 针尖指向针身的水平角度
            angle_v: 垂直角度（度）- 针尖指向针身的垂直角度
            body_length: 针身长度
            body_radius: 针身半径
            collision_mode: 碰撞检测模式
        """
        self.tip_position = tip if tip is not None else Point3Df()
        self.angle_horizontal = angle_h
        self.angle_vertical = angle_v
        self.body_length = body_length
        self.body_radius = body_radius
        self.collision_mode = collision_mode
        self.base_position = Point3Df()
        self.update_base_position()
    
    def update_base_position(self):
        """更新针身末端位置"""
        # 计算从针尖指向针身末端的方向
        if self.collision_mode == CollisionDetectionMode.Mode2D:
            direction = GeometryUtils.calculate_needle_direction_2d(self.angle_horizontal)
        else:
            direction = GeometryUtils.calculate_needle_direction_3d(self.angle_horizontal, self.angle_vertical)
        
        # 针身末端 = 针尖 + 方向 * 长度
        self.base_position = self.tip_position + direction * self.body_length
    
    def collides_with(self, other: 'NeedleGeometry') -> bool:
        """检查与另一个针的碰撞"""
        return GeometryUtils.capsule_capsule_collision(
            self.tip_position, self.base_position, self.body_radius,
            other.tip_position, other.base_position, other.body_radius
        )
    
    def collides_with_sphere(self, center: Point3Df, radius: float) -> bool:
        """检查与球形障碍物的碰撞"""
        return GeometryUtils.capsule_sphere_collision(
            self.tip_position, self.base_position, self.body_radius, center, radius
        )
    
    def collides_with_aabb(self, box_center: Point3Df, box_half_size: Point3Df) -> bool:
        """检查与AABB盒的碰撞"""
        return GeometryUtils.capsule_aabb_collision(
            self.tip_position, self.base_position, self.body_radius, box_center, box_half_size
        )
    
    def get_point_at(self, t: float) -> Point3Df:
        """获取针身上指定参数位置的点 (t=0: tip, t=1: base)"""
        return Point3Df.lerp(self.tip_position, self.base_position, t)
    
    def get_length(self) -> float:
        """获取针身长度"""
        return self.tip_position.distance(self.base_position)
    
    def get_direction(self) -> Point3Df:
        """获取针身方向（从针尖到针身末端）"""
        direction = self.base_position - self.tip_position
        length = direction.length()
        return direction * (1.0 / length) if length > 1e-6 else Point3Df(1, 0, 0)


