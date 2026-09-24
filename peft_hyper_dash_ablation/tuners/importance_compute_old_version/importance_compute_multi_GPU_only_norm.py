# coding=utf-8

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis

class HierarchicalImportanceAnalyzer:
    
    def __init__(self):
        # 虽然只用 Norm 打分，但保留 K/E 计算以备后续分析或日志记�?
        self.metrics = {
            'kurtosis': self._compute_kurtosis,
            'entropy': self._compute_entropy,
            'norm': self._compute_norm 
        }
        self.available_gpus = self._detect_available_gpus()
    
    def _detect_available_gpus(self):
        if not torch.cuda.is_available(): return []
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                x = torch.randn(10, 10, device=f'cuda:{i}') 
                del x
                available_gpus.append(i)
            except Exception: pass
        return available_gpus
    
    def analyze_model_only_target_modules(self, model, target_modules, target_module_keys, top_k=56,ratio=2):
        """
        top_k: 传入的总预�? (例如 56)
        """
        num_gpus = len(self.available_gpus)
        if num_gpus == 0:
            return self._analyze_cpu(model, target_module_keys, top_k)
        elif len(target_module_keys) < 20 or num_gpus == 1:
            return self._analyze_single_gpu(model, target_module_keys, top_k)
        else:
            return self._analyze_multi_gpu(model, target_modules, target_module_keys, top_k,ratio)

    def _compute_final_scores(self, raw_metrics_dict):
        """
        === 修改�? 1: �? Norm 打分公式 ===
        目标: Norm 越大 -> 更新幅度越大 -> 越不稳定 -> Score 应越�? (排前�?)
        """
        grouped = {}
        # 自动识别类型
        all_types = set()
        for name in raw_metrics_dict:
            for pt in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
                if pt in name: all_types.add(pt); break
        
        for pt in all_types:
            grouped[pt] = {'k': [], 'e': [], 'n': [], 'names': []}

        for name, m in raw_metrics_dict.items():
            for pt in all_types:
                if pt in name:
                    grouped[pt]['k'].append(m['kurtosis'])
                    grouped[pt]['e'].append(m['entropy'])
                    grouped[pt]['n'].append(m['norm'])
                    grouped[pt]['names'].append(name)
                    break
        
        final_scores = {}
        
        for pt, data in grouped.items():
            if not data['names']: continue
            
            # 虽然我们只用 Norm，但为了归一化，还是需要提取出�?
            n = np.array(data['n'])
            
            # Min-Max 归一�?
            def minmax(arr):
                _min, _max = arr.min(), arr.max()
                if _max - _min < 1e-6: return np.zeros_like(arr) + 0.5
                return (arr - _min) / (_max - _min)

            norm_n = minmax(n)
            
            # === 核心公式修改 ===
            # 只考虑 Norm。Norm 越大(norm_n接近1)，Score越小(接近0)�?
            # 这样在升序排序时，Norm 大的层会排在最前面 (Rank 1, 2, 3...)
            scores = 1.0 - norm_n
            
            for i, name in enumerate(data['names']):
                final_scores[name] = scores[i]
                
        return final_scores
    
    def _select_asymmetric_split(self, all_scores, total_budget):
        """
        === 修改�? 2: 具体的非对称分配 ===
        Total Budget (top_k) = 56
        Text: 100% Budget (56 layers)
        Others: 50% Budget (28 layers)
        """
        # 1. 排序：分数从低到�?
        # Low Score = High Norm = Unstable = Needs Exclusive
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1])
        all_modules_sorted = [x[0] for x in sorted_items]
        
        # 2. 计算截断�?
        # Text 独享 Top-56 (�? total_budget)
        limit_text = total_budget
        
        # Others 独享 Top-28 (�? total_budget 的一�?)
        limit_others = int(total_budget * 0.5)
        
        # 边界保护
        limit_text = min(max(1, limit_text), len(all_modules_sorted))
        limit_others = min(max(0, limit_others), len(all_modules_sorted))
        
        # 3. 生成名单
        text_exclusive = all_modules_sorted[:limit_text]     # �? 56 �?
        others_exclusive = all_modules_sorted[:limit_others] # �? 28 �? (是Text的子�?)
        
        print(f"\n{'='*60}")
        print(f"NORM-DRIVEN ASYMMETRIC ALLOCATION")
        print(f"Strategy: Norm-Only Score (High Norm = Rank 1)")
        print(f"{'='*60}")
        print(f"Total Budget (Top-K): {total_budget}")
        print(f"Text Exclusive:   Top {len(text_exclusive)} (100% of budget)")
        print(f"Others Exclusive: Top {len(others_exclusive)} (50% of budget)")
        print(f"Overlap: Others are a strict subset of Text exclusive layers.")
        
        return {
            "text_exclusive": text_exclusive,
            "others_exclusive": others_exclusive,
            "scores": all_scores
        }
    

    # --- GPU/Multi-GPU Wrappers ---
    def _analyze_multi_gpu(self, model, projection_types, target_module_keys, top_k):
        valid_modules = []
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight'): valid_modules.append((name, module.weight.data))
            except: continue

        chunks = self._split_into_chunks(valid_modules, len(self.available_gpus))
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(self.available_gpus)) as pool:
            results = []
            for i, chunk in enumerate(chunks):
                if chunk:
                    results.append(pool.apply_async(self._analyze_chunk_gpu_raw, (chunk, self.available_gpus[i])))
            
            all_raw = {}
            for res in results:
                try: all_raw.update(res.get(timeout=300))
                except: pass

        all_scores = self._compute_final_scores(all_raw)
        return self._select_asymmetric_split(all_scores, top_k)
        # return self._select_asymmetric_split_dynamic(all_scores, top_k,ratio)

    def _analyze_single_gpu(self, model, target_module_keys, top_k):
        device = torch.device(f'cuda:{self.available_gpus[0]}')
        all_raw = {}
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                w_gpu = module.weight.data.to(device)
                all_raw[name] = {
                    'kurtosis': self._compute_kurtosis_torch(w_gpu),
                    'entropy': self._compute_entropy_torch(w_gpu),
                    'norm': self._compute_norm_torch(w_gpu)
                }
            except: continue
        
        all_scores = self._compute_final_scores(all_raw)
        return self._select_asymmetric_split(all_scores, top_k)
    
    def _analyze_cpu(self, model, target_module_keys, top_k):
        all_raw = {}
        for name in target_module_keys:
            try:
                w = model.get_submodule(name).weight.data
                all_raw[name] = {
                    'kurtosis': self._compute_kurtosis(w),
                    'entropy': self._compute_entropy(w),
                    'norm': self._compute_norm(w)
                }
            except: continue
        all_scores = self._compute_final_scores(all_raw)
        return self._select_asymmetric_split(all_scores, top_k)

    # --- Metric Computation Functions (保留不动) ---
    def _split_into_chunks(self, data, n):
        if n == 0: return [data]
        s = len(data) // n + 1
        return [data[i:i + s] for i in range(0, len(data), s)]
    
    def _analyze_chunk_gpu_raw(self, chunk, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        res = {}
        for name, weight in chunk:
            try:
                w = weight.to(device)
                res[name] = {
                    'kurtosis': self._compute_kurtosis_torch(w), 
                    'entropy': self._compute_entropy_torch(w),
                    'norm': self._compute_norm_torch(w)
                }
            except: res[name] = {'kurtosis':0,'entropy':0,'norm':0}
        return res

    def _compute_kurtosis(self, t): return 1/(1+np.exp(-scipy_kurtosis(t.cpu().numpy().flatten(), fisher=True)/5)) if t.std()>1e-8 else 0
    def _compute_entropy(self, t, bins=50):
        h,_ = np.histogram(t.cpu().numpy().flatten(), bins, density=True); h=h[h>0]; h/=h.sum()
        return scipy_entropy(h)/np.log(bins)
    def _compute_norm(self, t): return torch.norm(t.float()).item()

    def _compute_kurtosis_torch(self, t):
        if t.std()<1e-8: return 0
        k = ((t-t.mean())**4).mean()/(t.std()**4)-3
        return 1/(1+torch.exp(-k/5)).item()
    def _compute_entropy_torch(self, t, bins=50):
        if t.max()-t.min()<1e-8: return 0.5
        h = torch.histc(t, bins, min=t.min(), max=t.max()); h=h[h>0]; h/=h.sum()
        return (-torch.sum(h*torch.log(h+1e-8))/torch.log(torch.tensor(bins, device=t.device))).item()
    def _compute_norm_torch(self, t): return torch.norm(t.float()).item()