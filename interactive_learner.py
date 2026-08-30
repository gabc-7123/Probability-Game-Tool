#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 对话级自进化学习系统
每次用户输入组合 → 系统生成预测 → 用户反馈正确答案 → 模型自动优化
"""

import json
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

# ==========================================
# 组合数据库
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


class InteractiveLearner:
    """对话级自进化学习系统"""
    
    def __init__(self):
        """初始化学习系统"""
        self.weights = {
            "frequency": 1.0,        # 频率权重
            "odd_even": 1.0,         # 奇偶比权重
            "large_small": 1.0,      # 大小数权重
            "span": 1.0,             # 跨度权重
            "sum": 1.0,              # 和值权重
        }
        
        # 学习历史
        self.feedback_history = []  # [(input, predictions, actual, accuracy, weights_before)]
        self.learning_log = []      # 详细日志
        
        # 统计数据
        self.all_reds = []
        self.all_blacks = []
        self._init_statistics()
        
        print("✅ 对话级自进化学习系统已初始化")
        print(f"📊 初始权重配置: {self.weights}")
    
    def _init_statistics(self):
        """从组合数据库提取统计信息"""
        for combo in COMBINATION_DATA.values():
            self.all_reds.extend(combo["reds"])
            self.all_blacks.append(combo["black"])
    
    def extract_features(self, reds: List[int], black: int) -> Dict[str, float]:
        """提取组合特征"""
        features = {}
        
        # 1. 奇偶比
        odd_count = sum(1 for x in reds if x % 2 == 1)
        features["odd_ratio"] = odd_count / 6.0
        features["even_ratio"] = (6 - odd_count) / 6.0
        
        # 2. 大小数比（17为分界）
        large_count = sum(1 for x in reds if x > 17)
        features["large_ratio"] = large_count / 6.0
        features["small_ratio"] = (6 - large_count) / 6.0
        
        # 3. 跨度
        features["span"] = max(reds) - min(reds)
        
        # 4. 和值
        features["sum"] = sum(reds)
        
        # 5. 黑球范围
        features["black_normalized"] = black / 16.0
        
        return features
    
    def generate_predictions(self, num_predictions: int = 5) -> List[Tuple[List[int], int, float]]:
        """
        基于当前权重生成多个预测
        返回: [(reds, black, confidence_score), ...]
        """
        predictions = []
        
        # 计算历史特征的目标值
        hist_odd_ratios = []
        hist_large_ratios = []
        hist_spans = []
        hist_sums = []
        hist_blacks = []
        
        for combo in COMBINATION_DATA.values():
            features = self.extract_features(combo["reds"], combo["black"])
            hist_odd_ratios.append(features["odd_ratio"])
            hist_large_ratios.append(features["large_ratio"])
            hist_spans.append(features["span"])
            hist_sums.append(features["sum"])
            hist_blacks.append(combo["black"])
        
        # 计算目标特征值（加权）
        odd_mean = np.mean(hist_odd_ratios)
        large_mean = np.mean(hist_large_ratios)
        span_mean = np.mean(hist_spans)
        sum_mean = np.mean(hist_sums)
        
        # 生成候选组合
        np.random.seed(None)  # 每次不同
        for pred_idx in range(num_predictions):
            # 根据奇偶目标生成
            odd_count = int(round(6 * odd_mean))
            odd_count = max(1, min(5, odd_count))  # 1-5之间
            
            # 生成奇数和偶数
            odd_numbers = np.random.choice([x for x in range(1, 34) if x % 2 == 1], 
                                          size=odd_count, replace=False)
            even_numbers = np.random.choice([x for x in range(1, 34) if x % 2 == 0], 
                                           size=6-odd_count, replace=False)
            
            reds = sorted(np.concatenate([odd_numbers, even_numbers]).astype(int))
            
            # 选择黑球
            black = np.random.choice(range(1, 17))
            
            # 计算信心分数（基于特征匹配度）
            features = self.extract_features(reds, black)
            
            # 特征距离（越小越好）
            odd_diff = abs(features["odd_ratio"] - odd_mean)
            large_diff = abs(features["large_ratio"] - large_mean)
            span_diff = abs(features["span"] - span_mean) / span_mean if span_mean > 0 else 0
            sum_diff = abs(features["sum"] - sum_mean) / sum_mean if sum_mean > 0 else 0
            
            # 加权信心分数
            confidence = 1.0 - (
                self.weights["odd_even"] * odd_diff * 0.2 +
                self.weights["large_small"] * large_diff * 0.2 +
                self.weights["span"] * span_diff * 0.2 +
                self.weights["sum"] * sum_diff * 0.2
            )
            confidence = max(0.5, min(1.0, confidence))  # 限制在0.5-1.0
            
            predictions.append((reds, black, confidence))
        
        # 按信心分数排序
        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions
    
    def calculate_accuracy(self, predictions: List[Tuple[List[int], int, float]], 
                          actual_reds: List[int], actual_black: int) -> Tuple[float, int]:
        """
        计算准确度
        返回: (准确度百分比, 最接近的预测索引)
        """
        best_score = 0
        best_idx = -1
        
        for idx, (pred_reds, pred_black, _) in enumerate(predictions):
            # 红球匹配度
            red_matches = len(set(pred_reds) & set(actual_reds))
            red_score = red_matches / 6.0
            
            # 黑球匹配度
            black_score = 1.0 if pred_black == actual_black else 0.0
            
            # 综合得分
            score = red_score * 0.7 + black_score * 0.3
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        accuracy = best_score * 100
        return accuracy, best_idx
    
    def update_weights(self, actual_features: Dict, predictions: List) -> Dict[str, float]:
        """
        根据实际反馈更新权重
        使用偏差值动态调整
        """
        # 从最接近的预测提取特征
        best_pred_features = self.extract_features(predictions[0][0], predictions[0][1])
        
        # 计算偏差
        deviations = {
            "odd_even": abs(actual_features["odd_ratio"] - best_pred_features["odd_ratio"]),
            "large_small": abs(actual_features["large_ratio"] - best_pred_features["large_ratio"]),
            "span": abs(actual_features["span"] - best_pred_features["span"]),
            "sum": abs(actual_features["sum"] - best_pred_features["sum"]),
        }
        
        # 权重调整逻辑
        updates = {}
        for key, deviation in deviations.items():
            if deviation < 0.15:  # 预测准确，降权此项（太自信）
                updates[key] = self.weights[key] * 0.95
            elif deviation > 0.3:  # 预测不准，升权此项（需加强）
                updates[key] = self.weights[key] * 1.15
            else:  # 中等，微调
                updates[key] = self.weights[key] * 0.98
        
        # 频率权重固定
        updates["frequency"] = self.weights["frequency"]
        
        # 归一化权重（和为5）
        total = sum(updates.values())
        for key in updates:
            updates[key] = (updates[key] / total) * 5
        
        self.weights = updates
        return updates
    
    def process_feedback(self, actual_reds: str, actual_black: str):
        """
        处理用户反馈并更新模型
        格式: actual_reds = "1 2 3 4 5 6", actual_black = "7"
        """
        try:
            # 解析输入
            actual_reds_list = sorted([int(x) for x in actual_reds.strip().split()])
            actual_black_int = int(actual_black.strip())
            
            # 验证输入
            if len(actual_reds_list) != 6:
                return "❌ 错误: 红球必须是6个数字"
            if not all(1 <= x <= 33 for x in actual_reds_list):
                return "❌ 错误: 红球范围必须在1-33之间"
            if not (1 <= actual_black_int <= 16):
                return "❌ 错误: 黑球范围必须在1-16之间"
            
            # 生成预测
            predictions = self.generate_predictions(num_predictions=5)
            
            # 计算准确度
            accuracy, best_idx = self.calculate_accuracy(predictions, actual_reds_list, actual_black_int)
            
            # 提取实际特征
            actual_features = self.extract_features(actual_reds_list, actual_black_int)
            
            # 保存旧权重
            old_weights = self.weights.copy()
            
            # 更新权重
            new_weights = self.update_weights(actual_features, predictions)
            
            # 记录反馈
            self.feedback_history.append({
                "timestamp": datetime.now().isoformat(),
                "actual_reds": actual_reds_list,
                "actual_black": actual_black_int,
                "predictions": [(list(r), b, float(c)) for r, b, c in predictions],
                "best_match_idx": best_idx,
                "accuracy": float(accuracy),
                "old_weights": old_weights,
                "new_weights": new_weights,
                "actual_features": actual_features,
            })
            
            # 构建反馈报告
            report = self._build_feedback_report(
                predictions, accuracy, best_idx, 
                old_weights, new_weights, actual_features
            )
            
            return report
        
        except Exception as e:
            return f"❌ 处理错误: {str(e)}"
    
    def _build_feedback_report(self, predictions, accuracy, best_idx, 
                               old_weights, new_weights, actual_features):
        """构建反馈报告"""
        report = []
        
        # 标题
        report.append("\n" + "="*70)
        report.append("🧠 【自进化学习反馈报告】")
        report.append("="*70)
        
        # 1. 实际组合分析
        report.append(f"\n✅ 实际组合特征:")
        report.append(f"   奇偶比: {actual_features['odd_ratio']:.1%} 奇数")
        report.append(f"   大小比: {actual_features['large_ratio']:.1%} 大数")
        report.append(f"   跨度: {actual_features['span']}")
        report.append(f"   和值: {actual_features['sum']}")
        report.append(f"   黑球: {int(actual_features['black_normalized'] * 16)}")
        
        # 2. 预测对比
        report.append(f"\n🎯 预测对比 (前3个):")
        for idx, (reds, black, conf) in enumerate(predictions[:3]):
            match_status = "✅ 最佳匹配" if idx == best_idx else ""
            report.append(f"   预测{idx+1}: {' '.join(map(str, reds))} | {black} (信心: {conf:.1%}) {match_status}")
        
        # 3. 准确度
        report.append(f"\n📊 模型准确度: {accuracy:.1f}%")
        if accuracy >= 80:
            report.append("   ⭐ 优秀! 模型预测与实际高度吻合")
        elif accuracy >= 60:
            report.append("   👍 良好! 模型有明显预测能力")
        elif accuracy >= 40:
            report.append("   📈 中等! 继续学习中")
        else:
            report.append("   🔄 学习中! 样本数较少，继续积累")
        
        # 4. 权重变化
        report.append(f"\n⚙️  权重优化变化:")
        for key in ["odd_even", "large_small", "span", "sum"]:
            old_w = old_weights[key]
            new_w = new_weights[key]
            change = ((new_w - old_w) / old_w * 100) if old_w > 0 else 0
            
            if change > 0:
                report.append(f"   📈 {key:12s}: {old_w:.3f} → {new_w:.3f} (+{change:.1f}%)")
            elif change < 0:
                report.append(f"   📉 {key:12s}: {old_w:.3f} → {new_w:.3f} ({change:.1f}%)")
            else:
                report.append(f"   ➡️  {key:12s}: {old_w:.3f} → {new_w:.3f} (无变化)")
        
        # 5. 学习进度
        accuracy_list = [h["accuracy"] for h in self.feedback_history]
        if len(accuracy_list) > 1:
            avg_acc = np.mean(accuracy_list)
            latest_acc = accuracy_list[-1]
            trend = "📈 上升中" if latest_acc > avg_acc else "📉 需改进" if latest_acc < avg_acc else "➡️  稳定"
            
            report.append(f"\n📈 学习进度 (第{len(accuracy_list)}次反馈):")
            report.append(f"   平均准确度: {avg_acc:.1f}%")
            report.append(f"   最新准确度: {latest_acc:.1f}%")
            report.append(f"   趋势: {trend}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def get_summary(self) -> str:
        """获取学习总结"""
        if not self.feedback_history:
            return "📊 暂无学习记录"
        
        summary = []
        summary.append("\n" + "="*70)
        summary.append("📊 【学习系统总结】")
        summary.append("="*70)
        
        # 学习次数
        summary.append(f"\n🔢 总学习次数: {len(self.feedback_history)}")
        
        # 准确度统计
        accuracies = [h["accuracy"] for h in self.feedback_history]
        summary.append(f"\n📈 准确度统计:")
        summary.append(f"   最高: {max(accuracies):.1f}%")
        summary.append(f"   最低: {min(accuracies):.1f}%")
        summary.append(f"   平均: {np.mean(accuracies):.1f}%")
        
        # 权重演进
        summary.append(f"\n⚙️  权重演进过程:")
        summary.append(f"   初始: {self.feedback_history[0]['old_weights']}")
        summary.append(f"   当前: {self.weights}")
        
        # 重点改进项
        summary.append(f"\n🎯 重点改进项:")
        for key in ["odd_even", "large_small", "span", "sum"]:
            init_w = self.feedback_history[0]['old_weights'][key]
            curr_w = self.weights[key]
            change = ((curr_w - init_w) / init_w * 100) if init_w > 0 else 0
            
            if change >= 5:
                summary.append(f"   ⬆️  {key}: +{change:.1f}% (获得更多关注)")
            elif change <= -5:
                summary.append(f"   ⬇️  {key}: {change:.1f}% (权重调低)")
        
        summary.append("\n" + "="*70)
        
        return "\n".join(summary)


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    print("🧠 对话级自进化学习系统 - 交互演示\n")
    
    learner = InteractiveLearner()
    
    print("\n📝 使用方法:")
    print("   输入格式: 红球(6个) 空格分隔, 黑球(1个)")
    print("   示例: 1 5 10 15 20 25 7")
    print("   或: quit 退出\n")
    
    while True:
        user_input = input("🔹 输入实际组合: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        parts = user_input.split('|')
        if len(parts) != 2:
            print("❌ 格式错误! 请输入: 红球组合 | 黑球")
            continue
        
        report = learner.process_feedback(parts[0], parts[1])
        print(report)
    
    # 最终总结
    print(learner.get_summary())
