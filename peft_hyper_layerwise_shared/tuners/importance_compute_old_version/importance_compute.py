import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis


class HierarchicalImportanceAnalyzer: 
    def __init__(self):
        self.metrics = {
            'kurtosis': self._compute_kurtosis,
            'entropy': self._compute_entropy, 
            'sparsity': self._compute_sparsity,
            'effective_rank': self._compute_effective_rank
        }
    
    def analyze_model_only_target_modules(self, model, target_module_keys, top_k=10):
        """Analyze only target modules' importance and return top-k important layers"""
        importance_scores = {}
        
        # Only iterate through target modules, not all modules
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    
                    # Skip very small layers
                    if weight.numel() < 100:
                        continue
                        
                    scores = {}
                    
                    # Calculate various importance metrics
                    scores['kurtosis'] = self._compute_kurtosis(weight)
                    scores['entropy'] = self._compute_entropy(weight)  
                    scores['sparsity'] = self._compute_sparsity(weight)
                    
                    if weight.dim() >= 2:
                        scores['effective_rank'] = self._compute_effective_rank(weight)
                    else:
                        scores['effective_rank'] = 0.5
                    
                    # Comprehensive scoring (consistent with your previous design)
                    importance_scores[name] = (
                        0.3 * scores['kurtosis'] +
                        0.3 * (1 - scores['entropy']) + 
                        0.2 * scores['sparsity'] +
                        0.2 * scores['effective_rank']
                    )
                    print(f"Analyzed {name}: score = {importance_scores[name]:.4f}")
                    
            except Exception as e:
                print(f"Warning: Could not analyze module {name}: {e}")
                continue
        
        # Select top-k important layers
        if importance_scores:
            sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_layers = [layer_name for layer_name, score in sorted_layers[:top_k]]
            
            print(f"Successfully analyzed {len(importance_scores)} target modules")
            return top_k_layers, importance_scores
        else:
            print("Warning: No target modules could be analyzed, using all as important")
            # If analysis fails, treat all target modules as important layers
            return target_module_keys[:top_k], {name: 1.0 for name in target_module_keys}
    
    def _compute_kurtosis(self, tensor):
        if tensor.std() < 1e-8:
            return 0
        import numpy as np
        from scipy.stats import kurtosis
        tensor_np = tensor.cpu().numpy().flatten()
        kurt = kurtosis(tensor_np, fisher=True)
        return 1 / (1 + np.exp(-kurt / 5))
    
    def _compute_entropy(self, tensor, bins=50):
        import numpy as np
        from scipy.stats import entropy as scipy_entropy
        tensor_np = tensor.cpu().numpy().flatten()
        hist, _ = np.histogram(tensor_np, bins=bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent = scipy_entropy(hist)
        return ent / np.log(bins)
    
    def _compute_kurtosis_torch(self, tensor):
        if tensor.std() < 1e-8:
            return 0
        # ��GPU�ϼ���
        mean = tensor.mean()
        std = tensor.std()
        n = tensor.numel()
        # ��ȹ�ʽ: E[(X-��)^4] / ��^4 - 3
        fourth_moment = ((tensor - mean) ** 4).mean()
        kurt = fourth_moment / (std ** 4) - 3
        return 1 / (1 + torch.exp(-kurt / 5)).item()

    def _compute_entropy_torch(self, tensor, bins=50):
        # ��GPU�ϼ���ֱ��ͼ
        min_val = tensor.min()
        max_val = tensor.max()
        if (max_val - min_val) < 1e-8:
            return 0.5
        
        hist = torch.histc(tensor, bins=bins, min=min_val, max=max_val)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        # �ع�ʽ: -�� p_i * log(p_i)
        ent = -torch.sum(hist * torch.log(hist))
        max_ent = torch.log(torch.tensor(bins, device=tensor.device))
        return (ent / max_ent).item()

    def _compute_sparsity(self, tensor, threshold_ratio=0.01):
        abs_tensor = tensor.abs()
        threshold = threshold_ratio * abs_tensor.max()
        sparsity = (abs_tensor < threshold).float().mean()
        return sparsity.item()
    
    def _compute_effective_rank(self, tensor, threshold=0.01):
        if tensor.dim() < 2:
            return 0.5
        try:
            U, S, V = torch.svd(tensor.float())
            S = S / S.max()
            effective_rank = (S > threshold).sum().float() / len(S)
            return effective_rank.item()
        except:
            return 0.5