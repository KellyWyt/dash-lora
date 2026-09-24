import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis

class HierarchicalImportanceAnalyzer:
    
    def __init__(self):
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
    
    def analyze_model_only_target_modules(self, model, target_modules, target_module_keys, top_k=64):
        """
        top_k: 总的独享名额预算 (Total Budget for Exclusive Layers)
        """
        num_gpus = len(self.available_gpus)
        if num_gpus == 0:
            return self._analyze_cpu(model, target_module_keys, top_k)
        elif len(target_module_keys) < 20 or num_gpus == 1:
            return self._analyze_single_gpu(model, target_module_keys, top_k)
        else:
            return self._analyze_multi_gpu(model, target_modules, target_module_keys, top_k)

    def _compute_final_scores(self, raw_metrics_dict):
        """
        计算重要性分数。
        目标: 
          - Score 越低 = 越不稳定 = 需要独享 (Exclusive)
          - Score 越高 = 越稳定 = 可以共享 (Shared)
        公式逻辑:
          - Kurtosis (K): 越大越稳定 -> 正贡献
          - Entropy (E): 越大越稳定 -> 正贡献
          - Norm (N): 越大越不稳定 (更新幅度大) -> 负贡献 (用 1-N)
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
            
            k = np.array(data['k'])
            e = np.array(data['e'])
            n = np.array(data['n'])
            
            def minmax(arr):
                _min, _max = arr.min(), arr.max()
                if _max - _min < 1e-6: return np.zeros_like(arr) + 0.5
                return (arr - _min) / (_max - _min)

            norm_k = minmax(k)
            norm_e = minmax(e)
            norm_n = minmax(n)
            
            # === 修改点：结合 Norm 的公式 ===
            # 权重: 30% K, 30% E, 40% Norm
            # Norm 越大 -> Lora幅度越大 -> 越不稳定 -> 分数应越低
            scores = 0.3 * norm_k + 0.3 * norm_e + 0.4 * (1.0 - norm_n)
            
            for i, name in enumerate(data['names']):
                final_scores[name] = scores[i]
                
        return final_scores
    
    def _select_asymmetric_split(self, all_scores, total_budget):
        """
        === 修改点：非对称分配逻辑 ===
        根据 70% / 30% 的比例分配独享名额。
        """
        # 1. 排序：分数从低到高。
        # Lowest Score = Most Unstable = Most need for Exclusive
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1])
        all_modules_sorted = [x[0] for x in sorted_items]
        
        # 2. 计算预算
        # 确保至少有 1 个，且不超过总数
        limit_text = int(total_budget * 0.7)
        limit_others = int(total_budget * 0.3)
        
        limit_text = min(max(1, limit_text), len(all_modules_sorted))
        limit_others = min(max(0, limit_others), len(all_modules_sorted))
        
        # 3. 切片
        # Text 列表包含前 70% 预算的最不稳定层
        text_exclusive = all_modules_sorted[:limit_text]
        # Others 列表包含前 30% 预算的最不稳定层 (是 Text 列表的子集，只有最不稳定的才独享)
        others_exclusive = all_modules_sorted[:limit_others]
        
        print(f"\n{'='*60}")
        print(f"ASYMMETRIC IMPORTANCE ALLOCATION (Total Budget: {total_budget})")
        print(f"{'='*60}")
        print(f"Text Exclusive Modules  (Top ~70%): {len(text_exclusive)}")
        print(f"Others Exclusive Modules (Top ~30%): {len(others_exclusive)}")
        print(f"Overlap: All 'Others' modules are typically included in 'Text' modules")
        
        # 返回字典供 LoraConfig 使用
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

    # --- Metric Computation Functions (No changes needed except adding norm) ---
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