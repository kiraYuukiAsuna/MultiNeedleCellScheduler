from GeometryType import *
from enum import Enum
from typing import Tuple


class CollisionDetectionConfig:
    """碰撞检测配置"""
    
    def __init__(self, mode: CollisionDetectionMode = CollisionDetectionMode.Mode2D,
                 needle_tip_safe_distance: float = 80.0,
                 needle_body_safe_distance: float = 100.0):
        """
        Args:
            mode: 碰撞检测模式（2D或3D）
            needle_tip_safe_distance: 针尖安全距离（微米）
            needle_body_safe_distance: 针身安全距离（微米）
        """
        self.mode = mode
        self.needle_tip_safe_distance = needle_tip_safe_distance
        self.needle_body_safe_distance = needle_body_safe_distance


class CollisionType(Enum):
    """碰撞类型"""
    NONE = 0
    NEEDLE_TIP_TO_NEEDLE_TIP = 1
    NEEDLE_TIP_TO_NEEDLE_BODY = 2
    NEEDLE_BODY_TO_NEEDLE_BODY = 3


class CollisionDetector:
    """碰撞检测器"""
    
    def __init__(self, config: CollisionDetectionConfig):
        """
        Args:
            config: 碰撞检测配置
        """
        self.config = config
    
    def check_collide(self, needle1: NeedleGeometry, needle2: NeedleGeometry) -> Tuple[bool, CollisionType]:
        """检查两个针是否碰撞
        
        Args:
            needle1: 第一个针
            needle2: 第二个针
            
        Returns:
            (是否碰撞, 碰撞类型)
        """
        eps = 1e-3  # 用于判断参数是否在端点（tip）
        
        if self.config.mode == CollisionDetectionMode.Mode2D:
            # 在 2D 模式下，将 Z 和垂直角置零并更新底座坐标
            needle1_copy = self._copy_needle(needle1)
            needle2_copy = self._copy_needle(needle2)
            
            needle1_copy.collision_mode = CollisionDetectionMode.Mode2D
            needle2_copy.collision_mode = CollisionDetectionMode.Mode2D
            needle1_copy.angle_vertical = 0.0
            needle2_copy.angle_vertical = 0.0
            needle1_copy.tip_position.z = 0.0
            needle2_copy.tip_position.z = 0.0
            
            needle1_copy.update_base_position()
            needle2_copy.update_base_position()
            
            if self._check_needle_tip_collision(needle1_copy, needle2_copy):
                # 细分 tip 碰撞：tip-tip vs tip-body
                dist_2d, t1, t2 = GeometryUtils.segment_to_segment_distance(
                    needle1_copy.tip_position, needle1_copy.base_position,
                    needle2_copy.tip_position, needle2_copy.base_position
                )
                if t1 <= eps and t2 <= eps and dist_2d < self.config.needle_tip_safe_distance:
                    return True, CollisionType.NEEDLE_TIP_TO_NEEDLE_TIP
                return True, CollisionType.NEEDLE_TIP_TO_NEEDLE_BODY
            
            if self._check_needle_body_collision(needle1_copy, needle2_copy):
                return True, CollisionType.NEEDLE_BODY_TO_NEEDLE_BODY
            
            return False, CollisionType.NONE
        else:
            # 3D 模式保持原始三维几何
            needle1_copy = self._copy_needle(needle1)
            needle2_copy = self._copy_needle(needle2)
            
            needle1_copy.collision_mode = CollisionDetectionMode.Mode3D
            needle2_copy.collision_mode = CollisionDetectionMode.Mode3D
            needle1_copy.update_base_position()
            needle2_copy.update_base_position()
            
            if self._check_needle_tip_collision(needle1_copy, needle2_copy):
                # 细分 tip 碰撞：tip-tip vs tip-body
                dist_3d, tt1, tt2 = GeometryUtils.segment_to_segment_distance(
                    needle1_copy.tip_position, needle1_copy.base_position,
                    needle2_copy.tip_position, needle2_copy.base_position
                )
                if tt1 <= eps and tt2 <= eps and dist_3d < self.config.needle_tip_safe_distance:
                    return True, CollisionType.NEEDLE_TIP_TO_NEEDLE_TIP
                return True, CollisionType.NEEDLE_TIP_TO_NEEDLE_BODY
            
            if self._check_needle_body_collision(needle1_copy, needle2_copy):
                return True, CollisionType.NEEDLE_BODY_TO_NEEDLE_BODY
            
            return False, CollisionType.NONE
    
    def _copy_needle(self, needle: NeedleGeometry) -> NeedleGeometry:
        """复制针对象"""
        new_needle = NeedleGeometry(
            tip=Point3Df(needle.tip_position.x, needle.tip_position.y, needle.tip_position.z),
            angle_h=needle.angle_horizontal,
            angle_v=needle.angle_vertical,
            body_length=needle.body_length,
            body_radius=needle.body_radius,
            collision_mode=needle.collision_mode
        )
        return new_needle
    
    def _check_needle_tip_collision(self, needle1: NeedleGeometry, needle2: NeedleGeometry) -> bool:
        """检查针尖碰撞"""
        tip_distance_1_to_2 = GeometryUtils.point_to_segment_distance(
            needle1.tip_position, needle2.tip_position, needle2.base_position
        )
        
        tip_distance_2_to_1 = GeometryUtils.point_to_segment_distance(
            needle2.tip_position, needle1.tip_position, needle1.base_position
        )
        
        return (tip_distance_1_to_2 < self.config.needle_tip_safe_distance or
                tip_distance_2_to_1 < self.config.needle_tip_safe_distance)
    
    def _check_needle_body_collision(self, needle1: NeedleGeometry, needle2: NeedleGeometry) -> bool:
        """检查针身碰撞"""
        body_dist, t1, t2 = GeometryUtils.segment_to_segment_distance(
            needle1.tip_position, needle1.base_position,
            needle2.tip_position, needle2.base_position
        )
        
        return body_dist < self.config.needle_body_safe_distance


