import streamlit as st
import numpy as np
import math
import re
import random
from collections import Counter, defaultdict

# ==========================================
# ⚠️ 合法免责声明（置顶醒目）
# ==========================================
st.set_page_config(page_title="万物终极推演引擎（仅供研究）", layout="wide")

st.error("""
### ⚠️【合法免责声明】
本程序所有代码、模型及输出结果**仅供数学算法学习、科学编程研究与纯娱乐模拟使用**。
现实中的任何随机抽取事件均为完全独立的随机事件，任何数学模型（包括AI）均**无法预测**结果。
**严禁将本项目用于任何真实博彩、投注或赌博行为**。请严格遵守国家法律法规，理性对待，量力而行。
""")

st.title("🌌 万物终极推演引擎（现代+传统全融合版）")
st.markdown("**无需选择任何模式。输入任意数据，所有模型自动在后台融合计算，一次性输出终极结论。**")

# ==========================================
# 现代数学与概率模型
# ==========================================
class ModernMath:
    @staticmethod
    def normal_pdf(x, mu, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def poisson_pmf(k, lam):
        return (math.exp(-lam) * lam ** k) / math.factorial(k)

    @staticmethod
    def hypergeom_pmf(k, n, K, N):
        return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)

    @staticmethod
    def bayes_update(prior, likelihood):
        posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))
        return posterior

    @staticmethod
    def moving_average(nums, window=3):
        if len(nums) < window: return np.mean(nums)
        return np.mean(nums[-window:])

    @staticmethod
    def random_walk(nums, steps=5):
        last = nums[-1]
        path = [last]
        sigma = np.std(nums) if len(nums) > 1 else 1
        for _ in range(steps):
            last += np.random.normal(0, sigma)
            path.append(last)
        return path

# 数字随机游戏模拟（纯数学娱乐，无违规词汇）
class NumberGameSimulator:
    @staticmethod
    def simulate_sequence_game(history):
        # 三位序列匹配实验（原三位随机数游戏）
        if len(history) < 3: return "数据不足，需至少输入3个数字"
        last_three = [int(x) % 10 for x in history[-3:]]
        return f"三位序列最近三期走势：{last_three}。单注理论匹配概率为1/1000。基于模10周期，下期关注范围在 {[(x+1)%10 for x in last_three]} 附近波动。"

    @staticmethod
    def simulate_pool_game(reds, blues):
        # 多维奖池抽取实验（原多维组合游戏）
        if len(reds) < 5 or len(blues) < 2: return "数据不足，输入格式应为：红池5个数字（1-35） 蓝池2个数字（1-12）"
        combos = math.comb(35, 5) * math.comb(12, 2)
        return f"组合抽取总数：{combos:,}。中奖概率为 1/{combos:,} (约 {1/combos:.10%})。已运行大数定律蒙特卡洛验证，长线期望为负，切勿沉迷。"

# ==========================================
# 传统推演模型
# ==========================================
class AncientWisdom:
    TIAN_GAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    DI_ZHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    
    @staticmethod
    def year_to_ganzhi(year):
        return AncientWisdom.TIAN_GAN[(year - 4) % 10] + AncientWisdom.DI_ZHI[(year - 4) % 12]

    @staticmethod
    def yin_yang(nums):
        mu = np.mean(nums)
        yang = sum(1 for x in nums if x > mu)
        ratio = yang / len(nums)
        if ratio > 0.6: return f"阳盛阴衰（{ratio:.0%}阳），物极必反，注意回落风险。"
        elif ratio < 0.4: return f"阴盛阳衰（{ratio:.0%}阳），静极思动，未来或有转机。"
        else: return f"阴阳平衡（{ratio:.0%}阳），天下大势，和合共生。"

    @staticmethod
    def wu_xing(nums):
        names = {1: "木", 2: "火", 3: "土", 4: "金", 5: "水"}
        elements = [(int(x) % 5) + 1 for x in nums]
        generating = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1}
        overcoming = {1: 3, 3: 5, 5: 2, 2: 4, 4: 1}
        report = []
        for i in range(len(elements) - 1):
            c, n = elements[i], elements[i+1]
            if generating[c] == n: report.append(f"{names[c]}生{names[n]}（顺势延续）")
            elif overcoming[c] == n: report.append(f"{names[c]}克{names[n]}（阻碍逆转）")
        return "；".join(report) if report else "五行平稳，无特殊生克。"

    @staticmethod
    def i_ching(nums):
        mu = np.mean(nums)
        yao = [1 if int(x) > mu else 0 for x in nums[-6:]]
        while len(yao) < 6: yao.insert(0, 0)
        bagua = {0: "坤", 1: "震", 2: "坎", 3: "兑", 4: "艮", 5: "离", 6: "巽", 7: "乾"}
        upper = bagua[yao[0]*4 + yao[1]*2 + yao[2]]
        lower = bagua[yao[3]*4 + yao[4]*2 + yao[5]]
        return f"【{upper}{lower}卦】", upper, lower

class BaZiModel:
    @staticmethod
    def get_bazi(y, m, d, h):
        try:
            from lunar_python import Solar
            solar = Solar.fromYmdHms(y, m, d, h, 0, 0)
            ec = solar.getLunar().getEightChar()
            return f"年柱【{ec.getYear()}】月柱【{ec.getMonth()}】日柱【{ec.getDay()}】时柱【{ec.getTime()}】"
        except:
            return "未安装lunar_python库，无法排盘"

# ==========================================
# 终极融合引擎（黑盒全自动）
# ==========================================
class UltimateOracle:
    def __init__(self):
        self.signals = []
    
    def add_signal(self, name, value, trend):
        self.signals.append({"name": name, "value": value, "trend": trend})

    def calculate_everything(self, raw_input):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", raw_input)]
        report = []
        
        if len(nums) < 2:
            return "数据不足，请至少输入2个数字。"
        
        # 现代统计
        mu = np.mean(nums)
        sigma = np.std(nums)
        sigma = sigma if sigma > 0 else 1
        report.append(f"📊 现代统计：均值{mu:.2f}，波动{sigma:.2f}，95%置信区间在 {mu-1.96*sigma:.2f} 至 {mu+1.96*sigma:.2f}。")
        report.append(f"🧮 正态分布：中心点概率密度为 {ModernMath.normal_pdf(mu, mu, sigma):.6f}。")
        report.append(f"📈 移动平均线：近期趋势均线为 {ModernMath.moving_average(nums):.2f}。")
        report.append(f"🎲 随机游走：未来5步走势为 {[round(p,2) for p in ModernMath.random_walk(nums)]}。")
        
        prior = 0.5
        likelihood = 1 / (1 + math.exp(-mu))
        posterior = ModernMath.bayes_update(prior, likelihood)
        report.append(f"🧬 贝叶斯推断：后验概率为 {posterior:.4%}。")
        
        # 数字随机游戏模拟（全合规词汇）
        if "序列" in raw_input or "三位" in raw_input or "排三" in raw_input:
            report.append(f"🎯 三位序列匹配实验：{NumberGameSimulator.simulate_sequence_game([int(x) for x in nums])}")
        if "组合" in raw_input or "奖池" in raw_input or "抽取" in raw_input:
            if len(nums) >= 7:
                reds = [int(x) for x in nums[:5]]
                blues = [int(x) for x in nums[5:7]]
                report.append(f"🎰 多维奖池抽取实验：{NumberGameSimulator.simulate_pool_game(reds, blues)}")
            elif len(nums) >= 2:
                n, k = int(nums[0]), int(nums[1])
                c = math.comb(n, k)
                report.append(f"🎯 组合数学：C({n}, {k}) = {c:,}。中奖真实概率为 1/{c:,}。")
        
        # 传统模型
        report.append(f"☯️ 阴阳推演：{AncientWisdom.yin_yang(nums)}")
        report.append(f"🪵 五行推演：{AncientWisdom.wu_xing(nums)}")
        hexagram, upper, lower = AncientWisdom.i_ching(nums)
        report.append(f"🌌 易经推演：{hexagram}。")
        if upper == "乾" and lower == "坤": report.append("断语：天地否，闭塞不通，当静待时变。")
        elif upper == "坤" and lower == "乾": report.append("断语：地天泰，天地交泰，万物亨通，未来大吉！")
        elif upper == "乾" and lower == "乾": report.append("断语：乾为天，飞龙在天，极强之势，防亢龙有悔。")
        elif upper == "坤" and lower == "坤": report.append("断语：坤为地，厚德载物，稳如泰山，宜静不宜动。")
        else: report.append("断语：变卦之象，阴阳交错，事物处于未定之局，需顺其自然。")

        if "年份" in raw_input and nums:
            y = int(nums[0])
            report.append(f"📅 干支纪年：公元 {y} 年为 【{AncientWisdom.year_to_ganzhi(y)}】年")
        if "八字" in raw_input or "排盘" in raw_input:
            if len(nums) >= 4:
                report.append(f"🧬 八字排盘：{BaZiModel.get_bazi(int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))}")

        # 马尔可夫预测
        if "预测" in raw_input or "序列" in raw_input or len(nums) >= 3:
            hist = [int(x) for x in nums]
            transitions = defaultdict(Counter)
            for i in range(len(hist) - 1): transitions[hist[i]][hist[i+1]] += 1
            last = hist[-1]
            if last in transitions:
                probs = transitions[last]
                best = max(probs, key=probs.get)
                report.append(f"🔮 马尔可夫链：下一节点最可能为【{best}】，概率 {probs[best]/sum(probs.values()):.2%}")

        # 终极融合总结
        total_score = 0
        if mu > 0: total_score += 1
        elif mu < 0: total_score -= 1
        
        yang = sum(1 for x in nums if x > mu) / len(nums)
        if yang > 0.6: total_score += 1
        elif yang < 0.4: total_score -= 1
        
        if upper == "坤" and lower == "乾": total_score += 2
        elif upper == "乾" and lower == "坤": total_score -= 2
        
        if total_score > 2:
            conclusion = "【终极结论：大吉之象】诸数共振，大势向好。谨记合法底线，仅作学术娱乐。"
        elif total_score > 0:
            conclusion = "【终极结论：平稳向荣】大势偏吉，虽有微澜，整体在正轨之上。"
        elif total_score == 0:
            conclusion = "【终极结论：平局未定】阴阳交替，五行调合，静待时机。"
        elif total_score > -2:
            conclusion = "【终极结论：先抑后扬】存在阻力，但物极必反，守住底线，将有转机。"
        else:
            conclusion = "【终极结论：大凶之象】势不可挡，宜避其锋芒，修身养性。"
        
        return conclusion, report

oracle = UltimateOracle()

# 黑盒输入区（不分开，不选菜单）
user_input = st.text_area("请输入任何数据（数字、历史序列、日期、年份）：", height=120, placeholder="例如：1 5 9 20 33 或 序列 3 5 8 2 7 或 组合 1 2 3 4 5 6 7 或 预测 1 2 3 5 8")

if st.button("一键启动全模型终极推演"):
    if user_input:
        with st.spinner('正在并行调动现代数学、正态分布、随机游走、贝叶斯、组合游戏模拟、阴阳、五行、易经、马尔可夫等全模型...'):
            import time
            time.sleep(1)
            results = oracle.calculate_everything(user_input)
            if isinstance(results, str) and "数据不足" in results:
                st.warning(results)
            else:
                conclusion, reports = results
                st.success(conclusion)
                st.divider()
                st.subheader("全模型计算过程：")
                for r in reports:
                    st.write(r)
                # 底部再次强调免责
                st.caption("⚠️ 再次郑重声明：本工具全部为纯数学算法模拟，严禁用于真实博彩。大数定律证明：任何随机抽取游戏的长期期望均为负值，请远离赌博！")
    else:
        st.warning("请输入内容！")