import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# 自定义简单的多目标优化问题：针头调度问题
class NeedleSchedulingProblem(Problem):
    def __init__(self):
        # 3个针头的位置 (x, y) 坐标，共6个决策变量
        # 目标: 最小化总移动距离 和 最小化针头间的最大距离差异
        super().__init__(
            n_var=6,  # 3个针头，每个针头2个坐标(x, y)
            n_obj=2,  # 2个目标函数
            n_constr=0,  # 无约束
            xl=np.array([0, 0, 0, 0, 0, 0]),  # 下界
            xu=np.array([10, 10, 10, 10, 10, 10])  # 上界
        )
        # 目标位置（3个细胞位置）
        self.targets = np.array([[2, 2], [5, 8], [9, 3]])
    
    def _evaluate(self, X, out, *args, **kwargs):
        # X的形状: (pop_size, 6)
        # 重塑为 (pop_size, 3, 2) - 3个针头，每个有x,y坐标
        needles = X.reshape(-1, 3, 2)
        
        # 目标1: 最小化从针头到目标细胞的总距离
        distances = np.linalg.norm(needles - self.targets, axis=2)
        total_distance = np.sum(distances, axis=1)
        
        # 目标2: 最小化针头间距离的方差（使针头分布均匀）
        needle_distances = []
        for i in range(needles.shape[0]):
            dists = []
            for j in range(3):
                for k in range(j+1, 3):
                    d = np.linalg.norm(needles[i, j] - needles[i, k])
                    dists.append(d)
            needle_distances.append(np.var(dists))
        
        out["F"] = np.column_stack([total_distance, needle_distances])

# 创建问题实例
problem = NeedleSchedulingProblem()

# 配置 NSGA-II
algorithm = NSGA2(pop_size=100)

# 运行优化
res = minimize(
    problem,
    algorithm,
    ('n_gen', 40),
    seed=1,
    verbose=True
)

print("\n优化完成!")
print("帕累托前沿解的数量:", len(res.X))
print("\n最优解示例 (前3个):")
for i in range(min(3, len(res.X))):
    needles = res.X[i].reshape(3, 2)
    print(f"\n解 {i+1}:")
    print(f"  针头位置: {needles}")
    print(f"  目标值 [总距离, 分布方差]: {res.F[i]}")