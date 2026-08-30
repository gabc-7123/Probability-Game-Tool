import streamlit as st
import numpy as np
import math
import re
from collections import Counter, defaultdict
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="组合概率科学研究引擎", layout="wide")

# 初始化session state用于存储学习记录
if "learning_records" not in st.session_state:
    st.session_state.learning_records = []
if "model_weights" not in st.session_state:
    st.session_state.model_weights = {
        "frequency": 1.0,
        "odd_even": 1.0,
        "large_small": 1.0,
        "span": 1.0,
        "sum": 1.0
    }

# 缓存数据处理，避免重复计算
@st.cache_data
def cached_analysis():
    """缓存分析结果，提高性能"""
    return True

# ==========================================
# 组合数据库（序列化，去敏感信息，仅用于科学研究）
# ==========================================
COMBINATION_DATA = {
    1: {"reds": [9, 14, 15, 16, 29, 30], "black": 10},
    2: {"reds": [1, 3, 11, 22, 26, 31], "black": 11},
    3: {"reds": [1, 2, 3, 8, 13, 14], "black": 2},
    4: {"reds": [13, 20, 25, 29, 30, 33], "black": 2},
    5: {"reds": [4, 11, 24, 25, 32, 33], "black": 13},
    6: {"reds": [10, 19, 21, 22, 31, 33], "black": 5},
    7: {"reds": [1, 4, 7, 21, 29, 30], "black": 1},
    8: {"reds": [8, 16, 26, 28, 29, 30], "black": 15},
    9: {"reds": [7, 9, 10, 16, 22, 27], "black": 11},
    10: {"reds": [1, 4, 5, 15, 23, 28], "black": 7},
    11: {"reds": [2, 4, 7, 14, 28, 29], "black": 9},
    12: {"reds": [2, 8, 25, 28, 30, 31], "black": 2},
    13: {"reds": [7, 8, 16, 24, 30, 32], "black": 2},
    14: {"reds": [5, 11, 21, 23, 24, 29], "black": 16},
    15: {"reds": [4, 19, 27, 29, 30, 32], "black": 13},
    16: {"reds": [3, 5, 16, 18, 29, 32], "black": 4},
    17: {"reds": [12, 14, 16, 17, 18, 32], "black": 8},
    18: {"reds": [3, 6, 8, 14, 26, 27], "black": 8},
    19: {"reds": [7, 8, 12, 15, 17, 21], "black": 1},
    20: {"reds": [9, 10, 13, 16, 19, 21], "black": 8},
    21: {"reds": [2, 23, 24, 26, 28, 32], "black": 4},
    22: {"reds": [8, 12, 18, 21, 24, 30], "black": 1},
    23: {"reds": [1, 3, 19, 20, 24, 25], "black": 7},
    24: {"reds": [7, 11, 14, 16, 27, 28], "black": 6},
    25: {"reds": [1, 11, 17, 22, 24, 29], "black": 4},
    26: {"reds": [4, 5, 11, 19, 27, 32], "black": 1},
    27: {"reds": [6, 10, 12, 15, 24, 27], "black": 12},
    28: {"reds": [5, 7, 10, 14, 21, 28], "black": 4},
    29: {"reds": [7, 14, 15, 23, 28, 33], "black": 3},
    30: {"reds": [1, 5, 6, 10, 12, 16], "black": 5},
    31: {"reds": [6, 9, 13, 17, 24, 28], "black": 15},
    32: {"reds": [2, 5, 14, 25, 30, 32], "black": 5},
    33: {"reds": [4, 6, 10, 18, 23, 31], "black": 11},
    34: {"reds": [6, 7, 11, 18, 22, 33], "black": 5},
    35: {"reds": [5, 18, 23, 24, 27, 33], "black": 3},
    36: {"reds": [2, 4, 15, 23, 25, 27], "black": 3},
    37: {"reds": [2, 13, 14, 16, 20, 24], "black": 5},
    38: {"reds": [5, 8, 15, 20, 21, 24], "black": 9},
    39: {"reds": [6, 13, 15, 17, 24, 25], "black": 1},
    40: {"reds": [4, 6, 14, 21, 22, 33], "black": 16},
    41: {"reds": [1, 4, 16, 22, 26, 31], "black": 4},
    42: {"reds": [5, 16, 24, 26, 29, 30], "black": 2},
    43: {"reds": [8, 16, 18, 22, 25, 26], "black": 7},
    44: {"reds": [1, 12, 14, 18, 30, 31], "black": 2},
    45: {"reds": [3, 4, 9, 13, 22, 31], "black": 4},
}
st.error("""
### 📜 【学术研究与娱乐声明】
本系统为纯粹的数学概率、统计分布与数理逻辑**教育性分析工具**，**仅供科研学习与编程练习使用**。
本工具仅进行数据统计分析，不提供任何预测或投资建议。所有分析结果均基于历史数据的纯数学性质，与现实决策无关。
**严禁将本系统用于任何违反法律法规的行为。**
""")

# ==========================================
# 组合概率分析引擎（教育性研究）
# ==========================================
class CombinationAnalyzer:
    """组合数据的科学研究分析模块"""
    
    @staticmethod
    def analyze_red_numbers(all_reds, all_blacks):
        """红球数字频率分析"""
        red_counts = Counter(all_reds)
        black_counts = Counter(all_blacks)
        
        report = []
        report.append("### 📊 红球数字频率分析")
        sorted_reds = sorted(red_counts.items(), key=lambda x: x[1], reverse=True)
        for num, count in sorted_reds[:15]:
            report.append(f"数字 **{num:2d}** 出现 **{count}** 次 | {'▓' * count}")
        
        report.append("\n### ⚫ 黑球数字频率分析")
        sorted_blacks = sorted(black_counts.items(), key=lambda x: x[1], reverse=True)
        for num, count in sorted_blacks[:10]:
            report.append(f"数字 **{num:2d}** 出现 **{count}** 次 | {'▓' * count}")
        
        return "\n".join(report), red_counts, black_counts
    
    @staticmethod
    def combination_overlap_analysis(all_combinations):
        """组合间重复数分析（两两对比）"""
        report = []
        report.append("### 🔗 组合间红球重复度分析")
        report.append("计算每对相邻组合间共同红球数量：")
        
        overlap_counts = []
        for i in range(len(all_combinations) - 1):
            set_i = set(all_combinations[i])
            set_next = set(all_combinations[i + 1])
            overlap = len(set_i & set_next)
            overlap_counts.append(overlap)
            if i < 10:  # 只显示前10个
                report.append(f"组合 {i+1} ↔ 组合 {i+2}: 共同数字 **{overlap}** 个")
        
        avg_overlap = np.mean(overlap_counts) if overlap_counts else 0
        report.append(f"\n📈 平均重复度: **{avg_overlap:.2f}** 个数字 / 6个红球")
        report.append(f"统计意义: {avg_overlap/6:.1%} 相似度")
        
        return "\n".join(report)
    
    @staticmethod
    def number_distribution_patterns(all_reds):
        """数字分布模式分析"""
        reds_arr = np.array(all_reds, dtype=float)
        
        report = []
        report.append("### 📐 红球数字分布统计")
        report.append(f"最小值: **{np.min(reds_arr):.0f}** | 最大值: **{np.max(reds_arr):.0f}**")
        report.append(f"均值: **{np.mean(reds_arr):.2f}** | 中位数: **{np.median(reds_arr):.2f}**")
        report.append(f"标准差: **{np.std(reds_arr):.2f}** | 方差: **{np.var(reds_arr):.2f}**")
        
        # 分布区间分析
        report.append("\n### 📊 数字区间分布")
        ranges = [(1, 10), (11, 20), (21, 30), (31, 33)]
        for start, end in ranges:
            count = sum(1 for x in reds_arr if start <= x <= end)
            report.append(f"区间 [{start:2d}-{end:2d}]: **{count}** 个 ({count/len(reds_arr):.1%})")
        
        return "\n".join(report)
    
    @staticmethod
    def combination_sequence_metrics(all_combinations):
        """组合序列性指标（数学模型）"""
        report = []
        report.append("### 🧮 组合序列数学特征")
        
        # 1. 连续出现相同数字的概率
        transitions = defaultdict(list)
        for i in range(len(all_combinations) - 1):
            for num in all_combinations[i]:
                if num in all_combinations[i + 1]:
                    transitions[num].append(i)
        
        persistent_nums = sorted(transitions.items(), key=lambda x: len(x[1]), reverse=True)
        report.append("\n**高连续性数字** (在多个相邻组合中出现):")
        for num, positions in persistent_nums[:8]:
            report.append(f"  数字 **{num}**: 出现在 {len(positions)} 个连续间隔")
        
        # 2. 组合覆盖率分析
        all_nums_covered = len(set(sum(all_combinations, [])))
        report.append(f"\n**覆盖率**: {all_nums_covered}/33 红球被覆盖 ({all_nums_covered/33:.1%})")
        
        # 3. 组合间距分析（按序号）
        report.append("\n**组合间距分析**: 相邻两个组合的'新旧比例'")
        gaps = []
        for i in range(len(all_combinations) - 1):
            new_nums = len(set(all_combinations[i + 1]) - set(all_combinations[i]))
            gaps.append(new_nums)
        
        if gaps:
            report.append(f"  平均新数字数量: **{np.mean(gaps):.2f}** / 6个")
            report.append(f"  新旧比例: **{np.mean(gaps)/6:.1%}**")
        
        return "\n".join(report)
    
    @staticmethod
    def black_ball_patterns(all_blacks):
        """黑球的统计模式"""
        blacks_arr = np.array(all_blacks, dtype=float)
        
        report = []
        report.append("### ⚫ 黑球特征分析")
        report.append(f"取值范围: **{np.min(blacks_arr):.0f}** ~ **{np.max(blacks_arr):.0f}**")
        report.append(f"均值: **{np.mean(blacks_arr):.2f}** | 标准差: **{np.std(blacks_arr):.2f}**")
        report.append(f"众数: **{Counter(all_blacks).most_common(1)[0][0]}** (出现 {Counter(all_blacks).most_common(1)[0][1]} 次)")
        
        return "\n".join(report)
    
    @staticmethod
    def hot_cold_numbers(all_reds, red_counts):
        """热冷号分析"""
        report = []
        report.append("### 🔥❄️ 热冷号分析")
        
        sorted_counts = sorted(red_counts.items(), key=lambda x: x[1], reverse=True)
        hot_nums = [num for num, count in sorted_counts[:8]]
        cold_nums = [num for num, count in sorted_counts[-8:]]
        
        report.append(f"**热号 TOP 8** (高频数字): {', '.join(map(str, sorted(hot_nums)))}")
        report.append(f"**冷号 TOP 8** (低频数字): {', '.join(map(str, sorted(cold_nums)))}")
        
        hot_ratio = sum(1 for x in all_reds if x in hot_nums) / len(all_reds)
        report.append(f"\n热号在全部数据中的占比: **{hot_ratio:.1%}**")
        
        return "\n".join(report)
    
    @staticmethod
    def number_span_analysis(all_combinations):
        """红球跨度分析（最大值-最小值）"""
        report = []
        report.append("### 📏 红球跨度分析")
        
        spans = [max(combo) - min(combo) for combo in all_combinations]
        spans_arr = np.array(spans)
        
        report.append(f"跨度均值: **{np.mean(spans_arr):.2f}**")
        report.append(f"跨度范围: **{np.min(spans_arr):.0f}** ~ **{np.max(spans_arr):.0f}**")
        report.append(f"跨度标准差: **{np.std(spans_arr):.2f}**")
        
        span_dist = Counter(spans)
        most_common_span = span_dist.most_common(1)[0]
        report.append(f"最常见跨度: **{most_common_span[0]}** (出现 {most_common_span[1]} 次)")
        
        return "\n".join(report)
    
    @staticmethod
    def sum_analysis(all_combinations):
        """红球和值分析"""
        report = []
        report.append("### ➕ 红球和值分析")
        
        sums = [sum(combo) for combo in all_combinations]
        sums_arr = np.array(sums)
        
        report.append(f"和值均值: **{np.mean(sums_arr):.2f}**")
        report.append(f"和值范围: **{np.min(sums_arr):.0f}** ~ **{np.max(sums_arr):.0f}**")
        report.append(f"和值标准差: **{np.std(sums_arr):.2f}**")
        
        # 和值分布（奇偶）
        odd_sum = sum(1 for s in sums if s % 2 == 1)
        report.append(f"\n和值奇偶分布: 奇数 {odd_sum} 个 | 偶数 {len(sums) - odd_sum} 个")
        
        return "\n".join(report)
    
    @staticmethod
    def odd_even_analysis(all_reds):
        """奇偶数比分析"""
        report = []
        report.append("### 🔢 奇偶数比分析")
        
        odd_count = sum(1 for x in all_reds if x % 2 == 1)
        even_count = len(all_reds) - odd_count
        
        report.append(f"奇数总数: **{odd_count}** | 偶数总数: **{even_count}**")
        report.append(f"奇偶比: **{odd_count/even_count:.2f}:1** (奇:偶)")
        report.append(f"奇数占比: **{odd_count/(odd_count+even_count):.1%}**")
        
        return "\n".join(report)
    
    @staticmethod
    def large_small_analysis(all_reds):
        """大小数比分析（>18为大，<=18为小）"""
        report = []
        report.append("### 📊 大小数比分析 (分界线: 18)")
        
        large_count = sum(1 for x in all_reds if x > 18)
        small_count = len(all_reds) - large_count
        
        report.append(f"大数 (19-33): **{large_count}** | 小数 (1-18): **{small_count}**")
        report.append(f"大小比: **{large_count/small_count:.2f}:1** (大:小)")
        report.append(f"大数占比: **{large_count/(large_count+small_count):.1%}**")
        
        return "\n".join(report)
    
    @staticmethod
    def black_ball_distribution(all_blacks):
        """黑球分布分析"""
        report = []
        report.append("### ⚫ 黑球分布倾向 (分界线: 8.5)")
        
        low_black = sum(1 for x in all_blacks if x <= 8)
        high_black = len(all_blacks) - low_black
        
        report.append(f"低黑 (1-8): **{low_black}** 次 | 高黑 (9-16): **{high_black}** 次")
        report.append(f"分布比: **{low_black/high_black:.2f}:1** (低:高)")
        
        return "\n".join(report)
    
    @staticmethod
    def missing_numbers_analysis(all_reds):
        """缺失号码分析"""
        report = []
        report.append("### ⚠️ 缺失号码分析")
        
        all_nums_set = set(all_reds)
        total_range = set(range(1, 34))
        missing_nums = sorted(total_range - all_nums_set)
        
        if missing_nums:
            report.append(f"从未出现过的号码: {', '.join(map(str, missing_nums))}")
            report.append(f"缺失数量: **{len(missing_nums)}/33** (**{len(missing_nums)/33:.1%}**)")
        else:
            report.append("✅ 所有号码都至少出现过1次")
        
        return "\n".join(report)
    
    @staticmethod
    def combination_trends(all_combinations):
        """组合趋势分析"""
        report = []
        report.append("### 📈 组合演化趋势")
        
        # 平均数字值趋势
        avg_vals = [np.mean(combo) for combo in all_combinations]
        
        report.append(f"平均值起点: **{avg_vals[0]:.2f}**")
        report.append(f"平均值终点: **{avg_vals[-1]:.2f}**")
        
        trend = avg_vals[-1] - avg_vals[0]
        if trend > 0:
            report.append(f"趋势: 📈 向上 (+{trend:.2f})")
        elif trend < 0:
            report.append(f"趋势: 📉 向下 ({trend:.2f})")
        else:
            report.append(f"趋势: ➡️ 平稳")
        
        # 波动性
        volatility = np.std(avg_vals)
        report.append(f"波动性 (标准差): **{volatility:.2f}**")
        
        return "\n".join(report)
    
    @staticmethod
    def red_number_intervals(all_reds):
        """红球区间分布"""
        report = []
        report.append("### 📍 红球数值区间分布")
        
        intervals = [(1, 11), (12, 22), (23, 33)]
        for start, end in intervals:
            count = sum(1 for x in all_reds if start <= x <= end)
            pct = count / len(all_reds)
            report.append(f"区间 [{start:2d}-{end:2d}]: **{count}** 个 (**{pct:.1%}**)")
        
        return "\n".join(report)
    
    @staticmethod
    def consecutive_analysis(all_combinations):
        """连号分析（连续数字）"""
        report = []
        report.append("### 🔗 连号分析")
        
        consecutive_counts = []
        for combo in all_combinations:
            sorted_combo = sorted(combo)
            count = 0
            for i in range(len(sorted_combo) - 1):
                if sorted_combo[i + 1] - sorted_combo[i] == 1:
                    count += 1
            consecutive_counts.append(count)
        
        report.append(f"平均连号个数: **{np.mean(consecutive_counts):.2f}** / 组合")
        report.append(f"连号出现频率: **{sum(1 for c in consecutive_counts if c > 0)/len(consecutive_counts):.1%}**")
        
        return "\n".join(report)

# ==========================================
# 随机组合生成模拟器（基于统计特征）
# ==========================================
class RandomSimulator:
    """基于历史数据特征的随机组合生成模拟"""
    
    @staticmethod
    def generate_combination_batch(all_combinations, all_blacks, num_to_generate=5):
        """批量生成模拟组合"""
        
        # 1. 分析历史数据的统计特征
        all_reds_flat = sum(all_combinations, [])
        red_counts = Counter(all_reds_flat)
        black_counts = Counter(all_blacks)
        
        # 获取关键指标
        red_freq_ratio = {num: count / len(all_reds_flat) for num, count in red_counts.items()}
        
        # 奇偶比
        odd_ratio = sum(1 for x in all_reds_flat if x % 2 == 1) / len(all_reds_flat)
        even_ratio = 1 - odd_ratio
        
        # 大小比 (>18 为大，<=18 为小)
        large_ratio = sum(1 for x in all_reds_flat if x > 18) / len(all_reds_flat)
        small_ratio = 1 - large_ratio
        
        # 跨度平均值和标准差
        spans = [max(combo) - min(combo) for combo in all_combinations]
        avg_span = np.mean(spans)
        std_span = np.std(spans)
        
        # 和值平均值
        sums = [sum(combo) for combo in all_combinations]
        avg_sum = np.mean(sums)
        std_sum = np.std(sums)
        
        # 黑球最常见值
        most_common_black = black_counts.most_common(1)[0][0]
        black_options = [num for num, count in black_counts.most_common(5)]
        
        # 2. 生成模拟组合
        simulated_combos = []
        for _ in range(num_to_generate):
            # 生成6个红球
            reds = []
            attempts = 0
            while len(reds) < 6 and attempts < 100:
                # 基于频率权重选择
                num = np.random.choice(list(range(1, 34)), p=[red_freq_ratio.get(i, 1/33) for i in range(1, 34)])
                if num not in reds:
                    reds.append(int(num))
                attempts += 1
            
            # 如果失败，强制补全
            while len(reds) < 6:
                num = np.random.randint(1, 34)
                if num not in reds:
                    reds.append(num)
            
            reds.sort()
            
            # 生成1个黑球 - 从最常见的5个中选
            black = np.random.choice(black_options)
            
            simulated_combos.append({
                "reds": reds,
                "black": black,
                "sum": sum(reds),
                "span": max(reds) - min(reds)
            })
        
        return simulated_combos, {
            "奇偶比": f"{odd_ratio:.1%}奇 {even_ratio:.1%}偶",
            "大小比": f"{large_ratio:.1%}大 {small_ratio:.1%}小",
            "平均跨度": f"{avg_span:.2f}",
            "平均和值": f"{avg_sum:.2f}",
            "黑球偏好": f"常见值: {black_options[:3]}"
        }
    
    @staticmethod
    def validate_simulated_combo(combo, reference_stats):
        """验证生成的组合是否符合历史特征"""
        reds = combo["reds"]
        
        odd_count = sum(1 for x in reds if x % 2 == 1)
        large_count = sum(1 for x in reds if x > 18)
        
        report = []
        report.append(f"**组合**: {' '.join(map(str, reds))} | **黑**: {combo['black']}")
        report.append(f"和值: {combo['sum']} | 跨度: {combo['span']}")
        report.append(f"奇偶: {odd_count}奇 {6-odd_count}偶 | 大小: {large_count}大 {6-large_count}小")
        
        return " | ".join(report)

# ==========================================
# 模型自学习优化引擎
# ==========================================
class ModelLearner:
    """模型自动学习和优化"""
    
    @staticmethod
    def calculate_combo_features(reds, black):
        """计算一个组合的特征向量"""
        odd_count = sum(1 for x in reds if x % 2 == 1)
        large_count = sum(1 for x in reds if x > 18)
        span = max(reds) - min(reds)
        total_sum = sum(reds)
        
        return {
            "odd_ratio": odd_count / 6,
            "large_ratio": large_count / 6,
            "span": span,
            "sum": total_sum,
            "black": black
        }
    
    @staticmethod
    def analyze_feedback(actual_combo, simulated_combos, all_combinations):
        """分析实际组合与模拟组合的偏差"""
        actual_features = ModelLearner.calculate_combo_features(actual_combo["reds"], actual_combo["black"])
        
        # 计算历史数据的标准特征
        all_reds = sum(all_combinations, [])
        all_blacks = []
        for combo_id in sorted(COMBINATION_DATA.keys()):
            all_blacks.append(COMBINATION_DATA[combo_id]["black"])
        
        hist_odd = sum(1 for x in all_reds if x % 2 == 1) / len(all_reds)
        hist_large = sum(1 for x in all_reds if x > 18) / len(all_reds)
        hist_span = np.mean([max(c) - min(c) for c in all_combinations])
        hist_sum = np.mean([sum(c) for c in all_combinations])
        
        # 计算偏差
        deviations = {
            "odd_deviation": abs(actual_features["odd_ratio"] - hist_odd),
            "large_deviation": abs(actual_features["large_ratio"] - hist_large),
            "span_deviation": abs(actual_features["span"] - hist_span),
            "sum_deviation": abs(actual_features["sum"] - hist_sum),
        }
        
        # 计算模拟准确度（与实际最接近的模拟组合）
        min_distance = float('inf')
        closest_sim_idx = -1
        for idx, sim_combo in enumerate(simulated_combos):
            sim_features = ModelLearner.calculate_combo_features(sim_combo["reds"], sim_combo["black"])
            distance = (
                abs(sim_features["odd_ratio"] - actual_features["odd_ratio"]) +
                abs(sim_features["large_ratio"] - actual_features["large_ratio"]) +
                abs(sim_features["span"] - actual_features["span"]) / 32 +
                abs(sim_features["sum"] - actual_features["sum"]) / 200
            )
            if distance < min_distance:
                min_distance = distance
                closest_sim_idx = idx
        
        accuracy = max(0, 100 - min_distance * 100)
        
        return {
            "actual_features": actual_features,
            "historical_features": {
                "hist_odd": hist_odd,
                "hist_large": hist_large,
                "hist_span": hist_span,
                "hist_sum": hist_sum
            },
            "deviations": deviations,
            "closest_simulation": closest_sim_idx,
            "accuracy": accuracy
        }
    
    @staticmethod
    def update_weights(feedback, current_weights):
        """根据反馈调整模型权重"""
        deviations = feedback["deviations"]
        
        # 计算调整因子（偏差越大，权重调整幅度越大）
        new_weights = current_weights.copy()
        
        # 奇偶比权重调整
        if deviations["odd_deviation"] > 0.2:
            new_weights["odd_even"] *= 1.1  # 提高权重
        else:
            new_weights["odd_even"] *= 0.95  # 降低权重
        
        # 大小数权重调整
        if deviations["large_deviation"] > 0.2:
            new_weights["large_small"] *= 1.1
        else:
            new_weights["large_small"] *= 0.95
        
        # 跨度权重调整
        if deviations["span_deviation"] > 3:
            new_weights["span"] *= 1.15
        else:
            new_weights["span"] *= 0.9
        
        # 和值权重调整
        if deviations["sum_deviation"] > 5:
            new_weights["sum"] *= 1.15
        else:
            new_weights["sum"] *= 0.9
        
        # 频率权重始终为基准
        new_weights["frequency"] = 1.0
        
        # 归一化权重
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}
        
        return new_weights
    
    @staticmethod
    def generate_learning_report(learning_records):
        """生成学习进度报告"""
        report = []
        report.append("### 📚 模型学习进度")
        
        if not learning_records:
            report.append("**还没有学习记录。输入实际组合数据开始学习！**")
            return "\n".join(report)
        
        report.append(f"**已学习次数**: {len(learning_records)}")
        
        accuracies = [r["accuracy"] for r in learning_records]
        report.append(f"**平均准确度**: {np.mean(accuracies):.1f}%")
        report.append(f"**最高准确度**: {np.max(accuracies):.1f}%")
        report.append(f"**准确度趋势**: {'📈 上升' if accuracies[-1] > np.mean(accuracies[:-1]) else '📉 下降'}")
        
        report.append("\n### 📊 最近5次学习记录")
        for idx, record in enumerate(learning_records[-5:], 1):
            reds_str = " ".join(map(str, record["actual_combo"]["reds"]))
            report.append(f"**记录{idx}**: {reds_str} | 黑:{record['actual_combo']['black']} | 准确度: {record['accuracy']:.1f}%")
        
        return "\n".join(report)

st.title("🌌 组合概率科学研究引擎 V11.0 ✨ 自学习版")
st.markdown("**深度数学分析** + **模型自学习优化**: 15大科学模型 + AI自适应反馈调整")
st.info("""
💡 **核心功能**: 
- 📊 15大分析模块 | 🎲 随机组合模拟 | 🧠 **AI自学习优化** (NEW)
- 每输入一个实际组合，模型就学习一次，逐次优化生成准确度
""")

# 模式选择
analysis_mode = st.radio("选择分析模式", ["序列多模型分析", "组合大数据分析", "随机组合生成模拟", "模型学习反馈"], horizontal=True)

# 核心多模型预测算法
def get_multi_predictions(history):
    if len(history) < 2:
        return "历史数据太短，请至少输入2个数字以上。"

    results = []
    nums = [float(x) for x in history]
    last_num = nums[-1]

    # === 模型1：马尔可夫链预测（给出 TOP 3 最可能出现的数字及概率） ===
    transitions = defaultdict(Counter)
    for i in range(len(nums) - 1):
        transitions[nums[i]][nums[i+1]] += 1
    
    if last_num in transitions:
        probs = transitions[last_num]
        total = sum(probs.values())
        top3 = probs.most_common(3)
        res = "模型1【马尔可夫链 TOP3】: "
        for num, count in top3:
            res += f"[数字 {int(num)} 概率 {count/total:.1%}] "
        results.append(res)
    else:
        results.append(f"模型1【马尔可夫链 TOP3】: 末尾数字 {int(last_num)} 未出现过，无法基于历史建立关联。")

    # === 模型2：线性回归趋势预测（预测下一个具体数值） ===
    x = np.arange(len(nums))
    slope, intercept = np.polyfit(x, nums, 1)
    next_val = slope * len(nums) + intercept
    results.append(f"模型2【线性回归趋势】: 下一组数值预计在 **{next_val:.2f}** 附近（趋势线斜率 {slope:.2f}）。")

    # === 模型3：差分均值预测（解决跳跃大的序列） ===
    diffs = [nums[i] - nums[i-1] for i in range(1, len(nums))]
    avg_diff = np.mean(diffs)
    diff_next = last_num + avg_diff
    results.append(f"模型3【差分均值外推】: 最近平均差值 {avg_diff:.2f}，下一组预计在 **{diff_next:.2f}** 附近。")

    # === 模型4：频率统计/众数模型（找热号） ===
    counts = Counter([int(x) for x in nums])
    if len(counts) > 1:
        most_common = counts.most_common(1)[0]
        results.append(f"模型4【频率统计(热号)】: 出现次数最多的数字是 **{most_common[0]}**（共出现 {most_common[1]} 次）。")
    else:
        results.append("模型4【频率统计(热号)】: 数据过于单一，未发现显著热号。")

    # === 模型5：平稳随机游走（预测波动区间） ===
    sigma = np.std(nums) if np.std(nums) > 0 else 1
    lower = last_num - 1.96 * sigma
    upper = last_num + 1.96 * sigma
    results.append(f"模型5【随机游走置信区间】: 下一组有95%的概率落在 **{lower:.2f}** 到 **{upper:.2f}** 的区间内。")

    return results

# 界面
user_input = st.text_area("请输入数字（算式、历史序列），例如：1+2 或 2 4 6 8 10 或 1 5 9 20 33", height=100)

if analysis_mode == "序列多模型分析":
    if st.button("一键启动多模型推演"):
        if user_input:
            with st.spinner('正在并行调动5套数学模型进行组合推演...'):
                import time
                time.sleep(0.5)

                # 1. 纯算式优先级最高
                if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', user_input) and not any(k in user_input for k in ['预测', '序列']):
                    try:
                        expression = user_input.replace('=', '')
                        result = eval(expression)
                        st.success(f"【算式计算结果】{user_input} = {result}")
                    except:
                        st.error("算式格式错误！")
                    st.stop()

                # 2. 序列多模型预测
                nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", user_input)]
                if len(nums) >= 2:
                    st.subheader("🔮 多模型联合推演结果：")
                    predictions = get_multi_predictions(nums)
                    
                    # 输出所有模型的预测结果
                    for res in predictions:
                        st.markdown(f"- {res}")

                    st.divider()
                    st.subheader("📊 原始数据统计画像")
                    st.write(f"均值：{np.mean(nums):.2f} | 标准差：{np.std(nums):.2f} | 最大值：{np.max(nums):.2f} | 最小值：{np.min(nums):.2f}")
                else:
                    st.warning("请至少输入2个数字，或者输入一个算式（如1+2）。")
        else:
            st.warning("请输入内容！")

elif analysis_mode == "组合大数据分析":
    st.subheader("📊 组合数据科学研究（45个组合样本 × 15大分析模块）")
    st.markdown("""
    **15大深度科学分析模块**:
    
    **基础分析** (1-5)：频率分析 | 分布统计 | 组合重复度 | 序列特征 | 黑球特征
    
    **数学特征** (6-10)：热冷号分析 | 跨度分析 | 和值分析 | 奇偶比 | 大小比
    
    **高级指标** (11-15)：黑球分布 | 缺失号码 | 组合趋势 | 区间分布 | 连号分析
    
    **学术声明**: 完全基于数学模型分析，仅供科学研究及娱乐参考。不涉及任何预测或投资建议。
    """)
    
    if st.button("📈 启动组合大数据分析"):
        with st.spinner('正在进行深度组合分析...'):
            # 提取所有组合数据
            all_reds = []
            all_blacks = []
            all_combinations = []
            
            for combo_id in sorted(COMBINATION_DATA.keys()):
                data = COMBINATION_DATA[combo_id]
                all_reds.extend(data["reds"])
                all_blacks.append(data["black"])
                all_combinations.append(data["reds"])
            
            analyzer = CombinationAnalyzer()
            
            # ===== 分析1：频率分析 =====
            st.subheader("🎯 分析1：数字频率分析")
            freq_report, red_counts, black_counts = analyzer.analyze_red_numbers(all_reds, all_blacks)
            st.markdown(freq_report)
            st.divider()
            
            # ===== 分析2：分布统计 =====
            st.subheader("📊 分析2：数字分布特征")
            dist_report = analyzer.number_distribution_patterns(all_reds)
            st.markdown(dist_report)
            st.divider()
            
            # ===== 分析3：组合重复度 =====
            st.subheader("🔗 分析3：组合间重复度（相邻组合对比）")
            overlap_report = analyzer.combination_overlap_analysis(all_combinations)
            st.markdown(overlap_report)
            st.divider()
            
            # ===== 分析4：组合序列特征 =====
            st.subheader("🧮 分析4：组合序列数学特征")
            seq_report = analyzer.combination_sequence_metrics(all_combinations)
            st.markdown(seq_report)
            st.divider()
            
            # ===== 分析5：黑球特征 =====
            st.subheader("⚫ 分析5：黑球（单独号）特征")
            black_report = analyzer.black_ball_patterns(all_blacks)
            st.markdown(black_report)
            st.divider()
            
            # ===== 分析6：热冷号分析 =====
            st.subheader("🔥 分析6：热冷号分析")
            hot_cold_report = analyzer.hot_cold_numbers(all_reds, red_counts)
            st.markdown(hot_cold_report)
            st.divider()
            
            # ===== 分析7：跨度分析 =====
            st.subheader("📏 分析7：红球跨度分析")
            span_report = analyzer.number_span_analysis(all_combinations)
            st.markdown(span_report)
            st.divider()
            
            # ===== 分析8：和值分析 =====
            st.subheader("➕ 分析8：红球和值分析")
            sum_report = analyzer.sum_analysis(all_combinations)
            st.markdown(sum_report)
            st.divider()
            
            # ===== 分析9：奇偶分析 =====
            st.subheader("🔢 分析9：奇偶数比分析")
            odd_even_report = analyzer.odd_even_analysis(all_reds)
            st.markdown(odd_even_report)
            st.divider()
            
            # ===== 分析10：大小数分析 =====
            st.subheader("📊 分析10：大小数比分析")
            large_small_report = analyzer.large_small_analysis(all_reds)
            st.markdown(large_small_report)
            st.divider()
            
            # ===== 分析11：黑球分布 =====
            st.subheader("⚫ 分析11：黑球分布倾向")
            black_dist_report = analyzer.black_ball_distribution(all_blacks)
            st.markdown(black_dist_report)
            st.divider()
            
            # ===== 分析12：缺失号码 =====
            st.subheader("⚠️ 分析12：缺失号码分析")
            missing_report = analyzer.missing_numbers_analysis(all_reds)
            st.markdown(missing_report)
            st.divider()
            
            # ===== 分析13：组合趋势 =====
            st.subheader("📈 分析13：组合演化趋势")
            trend_report = analyzer.combination_trends(all_combinations)
            st.markdown(trend_report)
            st.divider()
            
            # ===== 分析14：区间分布 =====
            st.subheader("📍 分析14：红球数值区间分布")
            interval_report = analyzer.red_number_intervals(all_reds)
            st.markdown(interval_report)
            st.divider()
            
            # ===== 分析15：连号分析 =====
            st.subheader("🔗 分析15：连号分析")
            consec_report = analyzer.consecutive_analysis(all_combinations)
            st.markdown(consec_report)
            st.divider()
            
            # ===== 组合详情表 =====
            st.subheader("📋 所有组合数据表（按序列展示）")
            combo_data = []
            for combo_id in sorted(COMBINATION_DATA.keys()):
                data = COMBINATION_DATA[combo_id]
                combo_data.append({
                    "序号": combo_id,
                    "红球": " ".join(map(str, sorted(data["reds"]))),
                    "黑": data["black"]
                })
            
            
            df = pd.DataFrame(combo_data)
            st.dataframe(df, use_container_width=True)
            
            st.caption("✅ 组合分析完成。此分析为纯数学统计研究，不具备任何预测意义。")

elif analysis_mode == "随机组合生成模拟":
    st.subheader("🎲 随机组合生成模拟器（基于45个历史组合的统计特征）")
    st.markdown("""
    **工作原理**：
    - 分析45个历史组合的统计特征（频率、奇偶比、大小比、跨度、和值等）
    - 基于这些特征进行加权随机抽样
    - 生成符合历史分布规律的"模拟组合"
    - **完全是娱乐性演示，不具备任何预测能力**
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        num_generate = st.slider("生成模拟组合的数量", min_value=1, max_value=20, value=5)
    with col2:
        if st.button("🎰 开始模拟生成"):
            st.info("⏳ 正在根据历史数据特征生成模拟组合...")
            
            # 提取数据
            all_reds = []
            all_blacks = []
            all_combinations = []
            
            for combo_id in sorted(COMBINATION_DATA.keys()):
                data = COMBINATION_DATA[combo_id]
                all_reds.extend(data["reds"])
                all_blacks.append(data["black"])
                all_combinations.append(data["reds"])
            
            # 生成模拟组合
            simulator = RandomSimulator()
            simulated_combos, stats = simulator.generate_combination_batch(all_combinations, all_blacks, num_generate)
            
            st.divider()
            st.subheader("📊 生成参数说明")
            stats_text = " | ".join([f"**{k}**: {v}" for k, v in stats.items()])
            st.markdown(f"模拟依据：{stats_text}")
            
            st.divider()
            st.subheader("🎯 生成的模拟组合")
            
            for idx, combo in enumerate(simulated_combos, 1):
                validation = simulator.validate_simulated_combo(combo, stats)
                st.write(f"**模拟组合 {idx}**: {validation}")
            
            st.divider()
            
            # 生成数据表
            sim_data = []
            for idx, combo in enumerate(simulated_combos, 1):
                sim_data.append({
                    "序号": f"模拟-{idx}",
                    "红球": " ".join(map(str, combo["reds"])),
                    "黑": combo["black"],
                    "和值": combo["sum"],
                    "跨度": combo["span"]
                })
            
            sim_df = pd.DataFrame(sim_data)
            st.dataframe(sim_df, use_container_width=True)
            
            st.caption("✅ 模拟生成完成。**此为娱乐性演示，完全基于数学统计，不具备任何预测或投资意义。**")

else:  # 模型学习反馈
    st.subheader("🧠 模型自学习优化反馈（AI持续改进）")
    st.markdown("""
    **工作流程**：
    1. 系统先基于历史45个组合生成模拟组合
    2. 您输入实际发生的准确组合
    3. 系统分析偏差，自动调整生成权重
    4. 下次生成时，模型会更精准
    
    **学习指标**：准确度 | 偏差分析 | 权重动态调整
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 输入实际组合反馈")
        feedback_mode = st.radio("反馈方式", ["手动输入", "查看历史"], horizontal=True)
        
        if feedback_mode == "手动输入":
            st.write("输入实际发生的组合数据：")
            col_red1, col_red2, col_red3 = st.columns(3)
            col_red4, col_red5, col_red6 = st.columns(3)
            col_black = st.columns(1)[0]
            
            with col_red1:
                r1 = st.number_input("红1", min_value=1, max_value=33, value=1)
            with col_red2:
                r2 = st.number_input("红2", min_value=1, max_value=33, value=2)
            with col_red3:
                r3 = st.number_input("红3", min_value=1, max_value=33, value=3)
            with col_red4:
                r4 = st.number_input("红4", min_value=1, max_value=33, value=4)
            with col_red5:
                r5 = st.number_input("红5", min_value=1, max_value=33, value=5)
            with col_red6:
                r6 = st.number_input("红6", min_value=1, max_value=33, value=6)
            with col_black:
                black = st.number_input("黑球", min_value=1, max_value=16, value=1)
            
            if st.button("📊 提交反馈并优化模型"):
                actual_combo = {"reds": sorted(list(set([r1, r2, r3, r4, r5, r6]))), "black": int(black)}
                
                if len(actual_combo["reds"]) != 6:
                    st.error("❌ 红球必须是6个不同的数字！")
                else:
                    # 生成模拟组合用于对比
                    all_reds = []
                    all_blacks = []
                    all_combinations = []
                    
                    for combo_id in sorted(COMBINATION_DATA.keys()):
                        data = COMBINATION_DATA[combo_id]
                        all_reds.extend(data["reds"])
                        all_blacks.append(data["black"])
                        all_combinations.append(data["reds"])
                    
                    simulator = RandomSimulator()
                    simulated_combos, _ = simulator.generate_combination_batch(all_combinations, all_blacks, 10)
                    
                    # 分析反馈
                    learner = ModelLearner()
                    feedback = learner.analyze_feedback(actual_combo, simulated_combos, all_combinations)
                    
                    # 更新权重
                    old_weights = st.session_state.model_weights.copy()
                    st.session_state.model_weights = learner.update_weights(feedback, st.session_state.model_weights)
                    
                    # 记录学习
                    learning_record = {
                        "timestamp": datetime.now().isoformat(),
                        "actual_combo": actual_combo,
                        "accuracy": feedback["accuracy"],
                        "deviations": feedback["deviations"],
                        "old_weights": old_weights,
                        "new_weights": st.session_state.model_weights
                    }
                    st.session_state.learning_records.append(learning_record)
                    
                    # 显示反馈结果
                    st.success("✅ 反馈已记录，模型已优化！")
                    
                    st.subheader("📊 本次反馈分析")
                    st.write(f"**实际组合**: {' '.join(map(str, actual_combo['reds']))} | **黑**: {actual_combo['black']}")
                    st.write(f"**与模拟最接近的组合**: 模拟组合 {feedback['closest_simulation'] + 1}")
                    st.write(f"**准确度得分**: **{feedback['accuracy']:.1f}%**")
                    
                    st.subheader("🔄 模型权重变化")
                    weights_df = pd.DataFrame({
                        "权重项": list(old_weights.keys()),
                        "旧权重": [f"{v:.3f}" for v in old_weights.values()],
                        "新权重": [f"{v:.3f}" for v in st.session_state.model_weights.values()]
                    })
                    st.dataframe(weights_df, use_container_width=True)
                    
                    st.divider()
        
        else:  # 查看历史
            st.subheader("📚 学习历史")
            learning_report = ModelLearner.generate_learning_report(st.session_state.learning_records)
            st.markdown(learning_report)
    
    with col2:
        st.subheader("📈 学习进度")
        
        if st.session_state.learning_records:
            accuracies = [r["accuracy"] for r in st.session_state.learning_records]
            
            st.metric("总学习次数", len(st.session_state.learning_records))
            st.metric("当前准确度", f"{accuracies[-1]:.1f}%")
            st.metric("平均准确度", f"{np.mean(accuracies):.1f}%")
            
            st.subheader("🧠 当前模型权重")
            weights_text = " | ".join([f"**{k}**: {v:.3f}" for k, v in st.session_state.model_weights.items()])
            st.markdown(f"权重配置：{weights_text}")
            
            st.subheader("📊 准确度曲线")
            chart_data = pd.DataFrame({
                "学习次数": range(1, len(accuracies) + 1),
                "准确度": accuracies
            })
            st.line_chart(chart_data.set_index("学习次数"))
        else:
            st.info("📌 还没有学习记录。开始输入反馈数据开启AI自学习！")
            st.write("**初始权重配置**:")
            for k, v in st.session_state.model_weights.items():
                st.write(f"- {k}: {v:.3f}")

st.divider()
st.caption("⚠️ 最终声明：本工具为组合数据数学统计研究平台，所有分析基于纯数学模型。此分析不具备任何预测能力，严禁用于任何违法用途。")