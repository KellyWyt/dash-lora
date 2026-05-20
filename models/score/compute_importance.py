import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy as scipy_entropy

class ParameterDistributionImportance:
    """通过参数分布分析层重要性的零成本方法"""
    
    def compute_importance_from_parameters(self, model):
        """
        仅通过分析参数分布来判断层重要性
        完全不需要前向传播！
        """
        importance_scores = {}

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                
                # 跳过很小的层
                if weight.numel() < 100:
                    continue
                    
                scores = {}

                # 1. 分布峰度（衡量分布的尖锐程度）
                scores['kurtosis'] = self.compute_kurtosis(weight)
                # 高峰度 → 分布不均匀 → 重要

                # 2. 熵（衡量分布的随机性）
                scores['entropy'] = self.compute_parameter_entropy(weight)
                # 低熵 → 分布有结构 → 重要

                # 3. 稀疏度（衡量参数的活跃程度）
                scores['sparsity'] = self.compute_sparsity(weight)
                # 高稀疏度 → 有选择性 → 重要

                # 4. 有效秩（矩阵的信息容量）
                if weight.dim() >= 2:
                    scores['effective_rank'] = self.compute_effective_rank(weight)
                else:
                    scores['effective_rank'] = 0.5

                # 综合评分
                importance_scores[name] = (
                    0.3 * scores['kurtosis'] +      # 峰度越高越重要
                    0.3 * (1 - scores['entropy']) + # 熵越低越重要  
                    0.2 * scores['sparsity'] +      # 稀疏度越高越重要
                    0.2 * scores['effective_rank']  # 有效秩越高越重要
                )

        return importance_scores

    def compute_kurtosis(self, tensor):
        """计算峰度 - 衡量分布是否比正态分布更尖或更平"""
        if tensor.std() < 1e-8:
            return 0
        tensor_np = tensor.cpu().numpy().flatten()
        from scipy.stats import kurtosis
        kurt = kurtosis(tensor_np, fisher=True)  # Fisher的定义，正态分布为0
        return 1 / (1 + np.exp(-kurt / 5))  # sigmoid归一化到0-1

    def compute_parameter_entropy(self, tensor, bins=50):
        """计算参数分布的熵"""
        tensor_np = tensor.cpu().numpy().flatten()
        hist, _ = np.histogram(tensor_np, bins=bins, density=True)
        hist = hist[hist > 0]  # 移除零值
        hist = hist / hist.sum()  # 归一化
        ent = scipy_entropy(hist)
        return ent / np.log(bins)  # 归一化到0-1

    def compute_sparsity(self, tensor, threshold_ratio=0.01):
        """计算参数的稀疏度"""
        abs_tensor = tensor.abs()
        threshold = threshold_ratio * abs_tensor.max()
        sparsity = (abs_tensor < threshold).float().mean()
        return sparsity.item()

    def compute_effective_rank(self, tensor, threshold=0.01):
        """通过奇异值计算有效秩"""
        if tensor.dim() < 2:
            return 0.5
            
        try:
            # 奇异值分解
            U, S, V = torch.svd(tensor.float())
            S = S / S.max()  # 归一化
            effective_rank = (S > threshold).sum().float() / len(S)
            return effective_rank.item()
        except:
            return 0.5