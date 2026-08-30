import streamlit as st
import numpy as np
import math
import re
from collections import Counter, defaultdict
from lunar_python import Solar

st.set_page_config(page_title="万物终极融合推演引擎", layout="wide")

st.title("🌌 万物终极融合推演引擎")
st.markdown("**输入任何数据，所有模型自动在后台融合计算，一次性输出终极结论。**")

# 内部算法引擎
class UltimateOracle:
    def __init__(self):
        self.signals = []  # 收集所有算法的信号

    def add_signal(self, name, value, trend):  # trend: -1(负面), 0(中性), 1(正面)
        self.signals.append({"name": name, "value": value, "trend": trend})

    def calculate_everything(self, raw_input):
        # 提取数字
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", raw_input)]
        if len(nums) < 2:
            return "数据不足，请至少输入2个数字，或者使用【年份 1990】、【八字 1990 5 8 12】等格式。"

        # 1. 现代统计学 (核心规律)
        mu = np.mean(nums)
        sigma = np.std(nums)
        trend = 0 if sigma == 0 else (1 if mu > 0 else -1)
        self.add_signal("现代统计学", f"均值{mu:.2f}，波动{sigma:.2f}", trend)
        
        # 2. 阴阳推演 (能量趋势)
        yang = sum(1 for x in nums if x > mu)
        ratio = yang / len(nums)
        if ratio > 0.6:
            self.add_signal("阴阳推演", "阳盛阴衰", 1)
        elif ratio < 0.4:
            self.add_signal("阴阳推演", "阴盛阳衰", -1)
        else:
            self.add_signal("阴阳推演", "阴阳平衡", 0)

        # 3. 五行推演 (生克转化)
        names = {1: "木", 2: "火", 3: "土", 4: "金", 5: "水"}
        elements = [(int(x) % 5) + 1 for x in nums]
        generating = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1}
        overcoming = {1: 3, 3: 5, 5: 2, 2: 4, 4: 1}
        wins = 0
        losses = 0
        for i in range(len(elements) - 1):
            c, n = elements[i], elements[i+1]
            if generating[c] == n: wins += 1
            elif overcoming[c] == n: losses += 1
        wuxing_trend = 1 if wins > losses else (-1 if losses > wins else 0)
        self.add_signal("五行推演", f"{wins}生{losses}克", wuxing_trend)

        # 4. 易经推演 (趋势吉凶)
        yao = [1 if int(x) > mu else 0 for x in nums[-6:]]
        while len(yao) < 6: yao.insert(0, 0)
        bagua = {0: "坤", 1: "震", 2: "坎", 3: "兑", 4: "艮", 5: "离", 6: "巽", 7: "乾"}
        upper = bagua[yao[0]*4 + yao[1]*2 + yao[2]]
        lower = bagua[yao[3]*4 + yao[4]*2 + yao[5]]
        
        hex_trend = 0
        gua_name = f"【{upper}{lower}卦】"
        if (upper == "乾" and lower == "坤") or (upper == "坎" and lower == "坎"):
            hex_trend = -1
        elif (upper == "坤" and lower == "乾") or (upper == "乾" and lower == "乾"):
            hex_trend = 1
        else:
            hex_trend = 0
        self.add_signal("易经推演", gua_name, hex_trend)

        # 5. 隐含的八字与年份
        if "年份" in raw_input and nums:
            y = int(nums[0])
            gan = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"][(y - 4) % 10]
            zhi = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"][(y - 4) % 12]
            self.add_signal("干支纪年", f"{gan}{zhi}年", 0)

        if "八字" in raw_input or "排盘" in raw_input:
            if len(nums) >= 4:
                solar = Solar.fromYmdHms(int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]), 0, 0)
                ec = solar.getLunar().getEightChar()
                self.add_signal("八字命盘", f"年{ec.getYear()}月{ec.getMonth()}日{ec.getDay()}时{ec.getTime()}", 0)

        # 6. 马尔可夫链预测 (未来走势)
        if len(nums) >= 3:
            hist = [int(x) for x in nums]
            transitions = defaultdict(Counter)
            for i in range(len(hist) - 1): transitions[hist[i]][hist[i+1]] += 1
            last = hist[-1]
            if last in transitions:
                probs = transitions[last]
                best = max(probs, key=probs.get)
                prob_val = probs[best]/sum(probs.values())
                if prob_val > 0.7:
                    self.add_signal("马尔可夫链", f"下一个最可能为【{best}】(概率{prob_val:.0%})", 1)
                elif prob_val > 0.5:
                    self.add_signal("马尔可夫链", f"下一个偏向【{best}】(概率{prob_val:.0%})", 0)
                else:
                    self.add_signal("马尔可夫链", f"下一节点变数很大，但【{best}】略高", -1)

        # 7. 组合数学 (如果提到抽奖)
        if "彩票" in raw_input or "组合" in raw_input:
            if len(nums) >= 2:
                n, k = int(nums[0]), int(nums[1])
                c = math.comb(n, k)
                self.add_signal("组合数学", f"C({n},{k})={c}，概率1/{c}", -1)

        # === 终极融合评分与结论 ===
        total_score = sum(s["trend"] for s in self.signals)
        if total_score > 2:
            conclusion = "【大吉之象】诸数共振，大势向好，宜乘势而上，果断行事。"
        elif total_score > 0:
            conclusion = "【平稳向荣】大势偏吉，虽有微澜，但整体在正轨之上。"
        elif total_score == 0:
            conclusion = "【平局未定】阴阳交替，五行调合，静待时机，顺势而为。"
        elif total_score > -2:
            conclusion = "【先抑后扬】存在阻力，但物极必反，守住底线，将有转机。"
        else:
            conclusion = "【大凶之象】势不可挡，宜避其锋芒，修身养性，静待冬天过去。"
        
        # 融合成一段话
        final_text = f"输入数据经全模型融合推演：{conclusion}\n\n【综合参数】："
        for s in self.signals:
            final_text += f"\n▪ {s['name']}：{s['value']}"
        
        return final_text

oracle = UltimateOracle()

# 界面（纯黑盒）
user_input = st.text_area("在这里输入任何数据（数字/日期/序列/年份）：", height=150, placeholder="例如：1 5 9 20 33 或 预测 1 2 3 5 8 或 年份 2024 或 八字 1990 5 8 12")
if st.button("启动终极融合计算"):
    if user_input:
        with st.spinner('正在调动所有模型并行推演...'):
            import time
            time.sleep(1)
            result = oracle.calculate_everything(user_input)
            if result.startswith("数据不足"):
                st.warning(result)
            else:
                st.success("推算完成！")
                st.code(result, language=None) # 直接用代码块形式呈现，因为是一整段话
    else:
        st.warning("请输入内容！")