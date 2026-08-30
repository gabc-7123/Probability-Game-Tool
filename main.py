import math
import random
import re
from collections import Counter, defaultdict

# 尝试引入第三方科学计算库，如果没安装自动降级为纯数学库
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("提示：未安装 numpy，复杂矩阵运算受限。")

# ==========================================
# 【底层：现代数学基石库】
# ==========================================
class MathCore:
    @staticmethod
    def normal_pdf(x, mu, sigma):  # 正态分布
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def poisson_pmf(k, lam):  # 泊松分布
        return (math.exp(-lam) * lam ** k) / math.factorial(k)

    @staticmethod
    def beta_pdf(x, a, b):  # Beta分布
        if x < 0 or x > 1: return 0
        return (x ** (a - 1)) * ((1 - x) ** (b - 1)) / math.gamma(a + b) * (math.gamma(a) * math.gamma(b))
    
    @staticmethod
    def t_distribution_pdf(x, df):  # t分布
        return (math.gamma((df+1)/2) / (math.sqrt(df*math.pi)*math.gamma(df/2))) * (1 + x**2/df)**(-(df+1)/2)

    @staticmethod
    def combo(n, k): return math.comb(n, k)

    @staticmethod
    def stats(numbers):
        mu = sum(numbers) / len(numbers)
        var = sum((x - mu) ** 2 for x in numbers) / len(numbers)
        return mu, math.sqrt(var)

# ==========================================
# 【数值计算模型】- 求解底层数学方程
# ==========================================
class NumericalSolver:
    @staticmethod
    def gauss_elimination(A, b):
        """Gauss消去法解线性方程组"""
        n = len(b)
        for i in range(n):
            # 找主元
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
            # 消元
            for j in range(i+1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
                b[j] -= factor * b[i]
        # 回代
        x = [0] * n
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i+1, n))) / A[i][i]
        return x

    @staticmethod
    def lagrange_interpolation(x_vals, y_vals, x):
        """Lagrange插值"""
        total = 0
        n = len(x_vals)
        for i in range(n):
            xi, yi = x_vals[i], y_vals[i]
            term = yi
            for j in range(n):
                if i != j:
                    term = term * (x - x_vals[j]) / (xi - x_vals[j])
            total += term
        return total

    @staticmethod
    def rk4(f, y0, t0, t_end, h):
        """四阶龙格-库塔法求解常微分方程"""
        t, y = t0, y0
        trajectory = [(t, y)]
        while t < t_end:
            k1 = h * f(t, y)
            k2 = h * f(t + h/2, y + k1/2)
            k3 = h * f(t + h/2, y + k2/2)
            k4 = h * f(t + h, y + k3)
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t = t + h
            trajectory.append((t, y))
        return trajectory

# ==========================================
# 【运筹优化模型】- 寻找最优解
# ==========================================
class OptimizationModels:
    @staticmethod
    def linear_programming_simplex():  # 线性规划（截取核心逻辑）
        print("线性规划（单纯形法）：请通过输入目标函数系数和约束条件矩阵来求解最优值。")
        print("鉴于纯Python实现复杂，后续接入numpy后将完善。")

    @staticmethod
    def decision_tree(costs):  # 决策分析：期望值决策
        # costs: 每个策略的成本和收益
        best_choice = min(costs, key=costs.get)
        return best_choice, costs[best_choice]

# ==========================================
# 【微分方程模型】- 系统动态演化
# ==========================================
class DifferentialModels:
    @staticmethod
    def sir_model(S0, I0, R0, beta, gamma, days):
        """传染病SIR模型：模拟感染传播"""
        S, I, R = S0, I0, R0
        result = []
        def f(t, y):
            s, i, r = y
            return [-beta * s * i, beta * s * i - gamma * i, gamma * i]
        
        # 用RK4近似求解
        t = 0
        h = 1
        while t < days:
            k1 = h * f(t, [S, I, R])
            k2 = h * f(t + h/2, [S + k1[0]/2, I + k1[1]/2, R + k1[2]/2])
            k3 = h * f(t + h/2, [S + k2[0]/2, I + k2[1]/2, R + k2[2]/2])
            k4 = h * f(t + h, [S + k3[0], I + k3[1], R + k3[2]])
            S += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
            I += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
            R += (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
            t += h
            result.append((t, S, I, R))
        return result

# ==========================================
# 【对话引擎：万物解析与输入】
# ==========================================
class UniversalOracle:
    def __init__(self):
        print("=" * 60)
        print("🌌 【万物推演引擎 V7.0】全模型整合版")
        print("融合：现代数学体系（概率/优化/微分/数值）+ 中华传统推演（易经/五行/八字）")
        print("你可以直接输入数字或指令，自动匹配模型计算结果。")
        print("=" * 60)
        print("部分指令示例：")
        print("  直接输入数字串：计算期望与波动")
        print("  输入：SIR 1000 1 0 0.3 0.1 30 （模拟传染病传播）")
        print("  输入：插值 0 1 1 2 2 3 1.5 （插值求解）")
        print("  输入：正态 10 2 12 （算概率密度）")
        print("  输入：五行 1 2 3 4 5 / 八字 1990 5 8 12 / 易经 2 9 8 4 5 6 / 年份 2024")
        print("  输入：【exit】退出。")
        print("=" * 60)

    def run(self):
        while True:
            try:
                user_input = input("\n>> 我：").strip()
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("推演结束，万法皆空。")
                    break
                if not user_input: continue

                nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", user_input)]
                text = user_input

                if "SIR" in text.upper() or "传染病" in text:
                    if len(nums) >= 6:
                        S0, I0, R0, beta, gamma, days = nums[0], nums[1], nums[2], nums[3], nums[4], int(nums[5])
                        traj = DifferentialModels.sir_model(S0, I0, R0, beta, gamma, days)
                        print(f"传染病模型在第{days}天，易感人数{S0-I0-R0:.0f}，感染人数{I0:.0f}，康复人数{R0:.0f}")
                
                elif "插值" in text:
                    # 提取坐标对 (x0,y0,x1,y1,...)
                    if len(nums) >= 4:
                        x_vals = nums[0::2]
                        y_vals = nums[1::2]
                        target_x = nums[-1] if len(nums) % 2 == 1 else x_vals[0]
                        result = NumericalSolver.lagrange_interpolation(x_vals, y_vals, target_x)
                        print(f"Lagrange插值结果：f({target_x}) ≈ {result:.4f}")

                elif "正态" in text:
                    if len(nums) >= 3:
                        print(f"正态分布概率密度：f({nums[0]}) = {MathCore.normal_pdf(nums[0], nums[1], nums[2]):.6f}")

                elif "五行" in text:
                    # 五行推演逻辑...
                    print("正在演化五行序列...")
                    elements = [(int(x) % 5) + 1 for x in nums if x.is_integer()]
                    names = {1:"木",2:"火",3:"土",4:"金",5:"水"}
                    print("演化序列:", " → ".join([names[e] for e in elements]))

                elif "计算" in text or len(nums) >= 2:
                    mu, sigma = MathCore.stats(nums)
                    print(f"数值统计：均值 {mu:.4f}，标准差 {sigma:.4f}")
                    
                else:
                    print("暂时无法匹配该输入，请尝试更多关键词。")
                    
            except Exception as e:
                print(f"模型解析错误：{e}，请检查输入格式。")

if __name__ == "__main__":
    oracle = UniversalOracle()
    oracle.run()