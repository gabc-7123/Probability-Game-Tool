import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from lunar_python import Solar

st.set_page_config(page_title="万物真相与奥义推演引擎", layout="wide")

st.title("🌌 万物真相与奥义推演引擎")
st.markdown("融合现代数学与东方智慧，输入数字或年份，计算万物运行规律。")

# 侧边栏设置
st.sidebar.header("推演模式选择")
mode = st.sidebar.selectbox("选择你想要的推演模式", 
    ["现代统计计算", "八字排盘", "易经推演", "五行生克", "组合概率"])

# 主界面输入区
user_input = st.text_area("请输入你的数字或数据（用空格分隔）：", placeholder="例如：1 5 9 20 33 或者 1990 5 8 12")
calculate_btn = st.button("开始推演")

if calculate_btn and user_input:
    try:
        # 提取数字
        import re
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", user_input)]
        
        if not nums:
            st.warning("请输入有效的数字！")
        else:
            if mode == "现代统计计算":
                if len(nums) < 2:
                    st.warning("统计计算需要至少两个数字！")
                else:
                    # 这里放入你的 V7.0 引擎里现代数学的逻辑
                    mu = np.mean(nums)
                    sigma = np.std(nums)
                    st.success(f"均值（真值）：{mu:.4f}")
                    st.success(f"标准差（波动）：{sigma:.4f}")
                    st.info("万事万物大概率波动在 " + f"{mu - sigma:.2f}" + " 至 " + f"{mu + sigma:.2f}" + " 之间。")
                    # 画个简单的分布图
                    fig, ax = plt.subplots()
                    ax.hist(nums, bins=10, color='#4A90E2', alpha=0.7)
                    ax.set_title('数字分布')
                    st.pyplot(fig)
            
            elif mode == "八字排盘":
                if len(nums) >= 4:
                    y, m, d, h = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
                    solar = Solar.fromYmdHms(y, m, d, h, 0, 0)
                    lunar = solar.getLunar()
                    ec = lunar.getEightChar()
                    st.success(f"你的八字排盘为：年柱【{ec.getYear()}】，月柱【{ec.getMonth()}】，日柱【{ec.getDay()}】，时柱【{ec.getTime()}】")
                else:
                    st.warning("八字排盘需要输入：公历年 月 日 时，例如：1990 5 8 12")
            
            elif mode == "易经推演":
                # 简单的易经逻辑，如需要可以接入更复杂的框架
                st.info("古法推演：根据输入数字的阴阳之象推演未来走势。")
                mu = np.mean(nums)
                yang = sum(1 for x in nums if x > mu)
                ratio = yang / len(nums)
                if ratio > 0.6:
                    st.warning("阳盛阴衰，物极必反，注意回落风险。")
                elif ratio < 0.4:
                    st.info("阴盛阳衰，静极思动，未来或有转机。")
                else:
                    st.success("阴阳平衡，中庸之道，平稳震荡。")

            elif mode == "五行生克":
                st.write("演化五行序列...")
                # 这里放入五行的逻辑

            elif mode == "组合概率":
                if len(nums) >= 2:
                    n, k = int(nums[0]), int(nums[1])
                    # 使用 math.comb
                    combos = math.comb(n, k)
                    st.success(f"从 {n} 个中选 {k} 个，组合数共有 {combos:,} 种。")
                    st.info(f"如果买一注，精确中奖概率为 1/{combos:,}。")
    except Exception as e:
        st.error(f"推演过程中出现了问题：{e}，请检查输入格式。")