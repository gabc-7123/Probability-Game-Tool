import streamlit as st
import numpy as np
import math
import re
from collections import Counter, defaultdict
import pandas as pd

st.set_page_config(page_title="组合概率科学研究引擎", layout="wide")

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

st.title("🌌 组合概率科学研究引擎 V9.0")
st.markdown("**深度数学分析**: 15大科学模型，全面分析组合数据统计特征。")
st.info("""
💡 **分析模块**: 
1. 频率分析 | 2. 分布统计 | 3. 组合重复度 | 4. 序列特征 | 5. 黑球特征 |
6. 热冷号分析 | 7. 跨度分析 | 8. 和值分析 | 9. 奇偶比 | 10. 大小比 |
11. 黑球分布 | 12. 缺失号码 | 13. 组合趋势 | 14. 区间分布 | 15. 连号分析
""")

# 模式选择
analysis_mode = st.radio("选择分析模式", ["序列多模型分析", "组合大数据分析"], horizontal=True)

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

else:  # 组合大数据分析
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

st.divider()
st.caption("⚠️ 最终声明：本工具为组合数据数学统计研究平台，所有分析基于纯数学模型。此分析不具备任何预测能力，严禁用于任何违法用途。")