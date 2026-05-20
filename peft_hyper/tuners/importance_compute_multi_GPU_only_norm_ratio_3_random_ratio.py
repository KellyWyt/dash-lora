# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis
import random

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
    
    def analyze_model_only_target_modules(self, model, target_modules, target_module_keys, top_k=56,ratio=2):
        num_gpus = len(self.available_gpus)
        if num_gpus == 0:
            return self._analyze_cpu(model, target_module_keys, top_k)
        elif len(target_module_keys) < 20 or num_gpus == 1:
            return self._analyze_single_gpu(model, target_module_keys, top_k)
        else:
            return self._analyze_multi_gpu(model, target_modules, target_module_keys, top_k,ratio)

    def _compute_final_scores(self, raw_metrics_dict):
        grouped = {}
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
            
            n = np.array(data['n'])
            
            def minmax(arr):
                _min, _max = arr.min(), arr.max()
                if _max - _min < 1e-6: return np.zeros_like(arr) + 0.5
                return (arr - _min) / (_max - _min)

            norm_n = minmax(n)
            
            scores = 1.0 - norm_n
            
            for i, name in enumerate(data['names']):
                final_scores[name] = scores[i]
                
        return final_scores
    
    def _select_asymmetric_split(self, all_scores, total_budget):

        # Low Score = High Norm = Unstable = Needs Exclusive
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1])
        all_modules_sorted = [x[0] for x in sorted_items]

        limit_text = total_budget
        
        limit_others = int(total_budget * 0.5)
        
        limit_text = min(max(1, limit_text), len(all_modules_sorted))
        limit_others = min(max(0, limit_others), len(all_modules_sorted))
        
        text_exclusive = all_modules_sorted[:limit_text]    
        others_exclusive = all_modules_sorted[:limit_others] 
        
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
    
    def _select_asymmetric_split_dynamic(self, all_scores, total_budget,ratio): # NOTE dash-lora

        # sorted_items = sorted(all_scores.items(), key=lambda x: x[1])

        import torch.distributed as dist
    
        # 获取当前进程的 Rank
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 原始排序逻辑
        # sorted_items = sorted(all_scores.items(), key=lambda x: x[1])

        # 在 sorted 之前，把所有卡的分数加起来取平均
        if dist.is_initialized():
            # 按照字母顺序对 key 排序，保证 tensor 转换时顺序一致
            names = sorted(all_scores.keys())
            score_tensor = torch.tensor([all_scores[n] for n in names], device='cuda')
            
            # 全局同步分数
            dist.all_reduce(score_tensor, op=dist.ReduceOp.AVG)
            
            # 重新写回 scores
            all_scores = {n: s.item() for n, s in zip(names, score_tensor)}

        # 此时再进行排序，所有卡得到的 Top 60 模块将物理上一模一样
        sorted_items = sorted(all_scores.items(), key=lambda x: (x[1], x[0]))
                
        # ========================== 调试打印开始 ==========================
        print(f"\n" + "!"*40)
        print(f"DISTRIBUTED RANK [{rank}] DEBUG - TOP 50 MODULES")
        print(f"Total modules detected: {len(all_scores)}")
        
        # 打印前 50 位，观察模块名和分数是否对齐
        for i, (name, score) in enumerate(sorted_items[:50]):
            # 注意：使用 :.8f 打印高精度分数，因为差异往往在末尾
            print(f"Rank {rank} | Pos {i:2}: {name:<40} | Score: {score:.8f}")
        
        print("!"*40 + "\n")
        # ========================== 调试打印结束 ==========================

        all_modules_sorted = [x[0] for x in sorted_items]
        max_layers = len(all_modules_sorted)
        

        actual_budget = total_budget * 2 
        

        ratio_ = ratio
        
        
        limit_others = int(actual_budget / (ratio_ + 2))
        limit_text = int(limit_others * ratio_)
        
        limit_text = min(max(1, limit_text), max_layers)
        limit_others = min(max(1, limit_others), limit_text) 
        
        text_exclusive = all_modules_sorted[:limit_text]      
        others_exclusive = all_modules_sorted[:limit_others]  
        
        print(f"\n{'='*60}")
        print(f"RANDOM RATIO DYNAMIC ALLOCATION")
        print(f"{'='*60}")
        print(f"Total Slot Budget: {actual_budget} (Derived from input {total_budget} * 2)")
        print(f"Random Ratio (Text/Others): {ratio:.4f}")
        print(f"-"*30)
        print(f"Calculated Text Budget:   Top {limit_text}")
        print(f"Calculated Others Budget: Top {limit_others}")
        print(f"Actual Ratio: {limit_text / limit_others:.2f} : 1 : 1")
        print(f"Total Slots Used: {limit_text + limit_others * 2}")
        
        return {
            "text_exclusive": text_exclusive,
            "others_exclusive": others_exclusive,
            "scores": all_scores,
            "stats": {
                "ratio": ratio_, 
                "text_limit": limit_text, 
                "others_limit": limit_others
            }
        }

    def _select_asymmetric_split_random_order(self, all_scores, total_budget, ratio):
        # 1. 不按分数排序，而是直接获取所有模块名并随机打乱
        # all_modules = list(all_scores.keys())
        all_modules = sorted(list(all_scores.keys()))
        # random.seed(42)  # multi-GPU,seed must be the same
        # random.shuffle(all_modules) 
        rng = random.Random(42) 
        rng.shuffle(all_modules)
        
        max_layers = len(all_modules)
        actual_budget = total_budget * 2 
        
        # 2. 保持原有的 Budget 计算逻辑
        limit_others = int(actual_budget / (ratio + 2))
        limit_text = int(limit_others * ratio)
        
        limit_text = min(max(1, limit_text), max_layers)
        limit_others = min(max(1, limit_others), limit_text) 
        
        # 3. 从随机打乱后的列表中提取
        text_exclusive = all_modules[:limit_text]      
        others_exclusive = all_modules[:limit_others]  
        
        print(f"--- ABLATION: RANDOM ORDER ---")
        print(f"Text Slots: {len(text_exclusive)}, Others Slots: {len(others_exclusive)}")
        
        return {
            "text_exclusive": text_exclusive,
            "others_exclusive": others_exclusive,
            "scores": all_scores,
            "stats": {"ratio": ratio, "text_limit": limit_text, "others_limit": limit_others, "mode": "random_order"}
        }

    def _select_ratio_by_gradient(self, all_scores, total_budget, num_steps=20, lr=0.1):
        """Use gradient descent on importance scores to find optimal modality ratios."""
        sorted_items = sorted(all_scores.items(), key=lambda x: (x[1], x[0]))
        all_modules_sorted = [x[0] for x in sorted_items]
        scores_tensor = torch.tensor([s for _, s in sorted_items], dtype=torch.float32)
        max_layers = len(all_modules_sorted)
        actual_budget = total_budget * 2

        # Learnable logits for 3 modalities (text, video, audio)
        ratio_logits = torch.zeros(3, requires_grad=True)
        optimizer = torch.optim.Adam([ratio_logits], lr=lr)

        for _ in range(num_steps):
            optimizer.zero_grad()
            ratios = torch.softmax(ratio_logits, dim=0)  # (a, b, c), sum=1, a dominates
            # Enforce text >= 0.5 via reparameterization: a = 0.5 + sigmoid(l0)*0.5
            a = 0.5 + torch.sigmoid(ratio_logits[0]) * 0.5
            bc = torch.softmax(ratio_logits[1:], dim=0) * (1.0 - a)
            b, c = bc[0], bc[1]

            limit_text = (actual_budget * a).clamp(1, max_layers)
            limit_video = (actual_budget * b).clamp(1, max_layers)
            limit_audio = (actual_budget * c).clamp(1, max_layers)

            # Soft selection: weighted sum of scores using sigmoid gates
            positions = torch.arange(max_layers, dtype=torch.float32)
            # Lower score = higher norm = more important to assign exclusive
            # We want to minimize the average score of selected layers
            loss = (
                (torch.sigmoid(limit_text  - positions) * scores_tensor).mean() +
                (torch.sigmoid(limit_video - positions) * scores_tensor).mean() +
                (torch.sigmoid(limit_audio - positions) * scores_tensor).mean()
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            a = (0.5 + torch.sigmoid(ratio_logits[0]) * 0.5).item()
            bc = torch.softmax(ratio_logits[1:], dim=0) * (1.0 - a)
            b, c = bc[0].item(), bc[1].item()

        limit_text  = min(max(1, int(actual_budget * a)), max_layers)
        limit_video = min(max(1, int(actual_budget * b)), max_layers)
        limit_audio = min(max(1, int(actual_budget * c)), max_layers)

        text_exclusive  = all_modules_sorted[:limit_text]
        video_exclusive = all_modules_sorted[:limit_video]
        audio_exclusive = all_modules_sorted[:limit_audio]

        print(f"\n{'='*60}")
        print(f"GRADIENT-INFERRED RATIO ALLOCATION (V-A-T)")
        print(f"{'='*60}")
        print(f"Inferred Ratios: Text={a:.4f}, Video={b:.4f}, Audio={c:.4f}")
        print(f"Calculated Limits: Text={limit_text}, Video={limit_video}, Audio={limit_audio}")
        print(f"Total Slots Used: {limit_text + limit_video + limit_audio}")

        return {
            "text_exclusive": text_exclusive,
            "video_exclusive": video_exclusive,
            "audio_exclusive": audio_exclusive,
            "scores": all_scores,
            "stats": {"ratios": (a, b, c), "limits": (limit_text, limit_video, limit_audio)}
        }

    def _select_asymmetric_split_random_ratio_actually(self, all_scores, total_budget, ratio): #NOTE compare experiment(order same,but random ratio)
        # name as tie-breaking key to guarantee deterministic order across all ranks
        sorted_items = sorted(all_scores.items(), key=lambda x: (x[1], x[0]))
        all_modules_sorted = [x[0] for x in sorted_items]
        max_layers = len(all_modules_sorted)

        actual_budget = total_budget * 2

        # rng = random.Random(42) #seed-42
        # rng = random.Random(123) #seed-123
        rng = random.Random(456) #seed-0


        eps = 0.02
        rem = 0.5 - (eps * 2) 
        r1 = rng.uniform(0, rem)
        r2 = rng.uniform(0, rem - r1)
        
        a = 0.5 + r1
        b = eps + r2
        c = 1.0 - a - b # c = eps + (rem - r1 - r2)
        
        # limit_others = int(actual_budget / (ratio_ + 2))
        # limit_text = int(limit_others * ratio_)
        
        # limit_text = min(max(1, limit_text), max_layers)
        # limit_others = min(max(1, limit_others), limit_text) 
        limit_text = int(actual_budget * a)
        limit_video = int(actual_budget * b)
        limit_audio = int(actual_budget * c)

        # 边界检查：确保不越界且至少为 1
        limit_text = min(max(1, limit_text), max_layers)
        limit_video = min(max(1, limit_video), max_layers)
        limit_audio = min(max(1, limit_audio), max_layers)
        
        text_exclusive = all_modules_sorted[:limit_text]
        video_exclusive = all_modules_sorted[:limit_video]
        audio_exclusive = all_modules_sorted[:limit_audio]
        
        print(f"\n{'='*60}")
        print(f"RANDOM RATIO DYNAMIC ALLOCATION (V-A-T)")
        print(f"{'='*60}")
        print(f"Generated Ratios: Text={a:.4f}, Video={b:.4f}, Audio={c:.4f}")
        print(f"Calculated Limits: Text={limit_text}, Video={limit_video}, Audio={limit_audio}")
        print(f"Total Slots Used: {limit_text + limit_video + limit_audio}")

        return {
            "text_exclusive": text_exclusive,
            "video_exclusive": video_exclusive,
            "audio_exclusive": audio_exclusive,
            "scores": all_scores,
            "stats": {
                "ratios": (a, b, c),
                "limits": (limit_text, limit_video, limit_audio)
            }
        }

    # --- GPU/Multi-GPU Wrappers ---
    def _analyze_multi_gpu(self, model, projection_types, target_module_keys, top_k,ratio):
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
                # except: pass
                except Exception as e:
                    print(f"[WARNING] chunk {i} failed: {e}")  # 不要静默pass

        all_scores = self._compute_final_scores(all_raw)
        # return self._select_asymmetric_split(all_scores, top_k)
        # return self._select_asymmetric_split_dynamic(all_scores, top_k,ratio)
    
        # return self._select_asymmetric_split_random_order(all_scores, top_k,ratio) #NOTE compare experiment(ratio same,but random order)
        return self._select_asymmetric_split_random_ratio_actually(all_scores, top_k,ratio) #NOTE compare experiment(order same,but random ratio)
        # return self._select_ratio_by_gradient(all_scores, top_k) #NOTE gradient-inferred ratio
    


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