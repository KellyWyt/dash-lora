import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis


class HierarchicalImportanceAnalyzer:
    
    def __init__(self):
        #移除 Rank 和 Sparsity，只保留有效的 Kurtosis 和 Entropy
        #NOTE 添加 norm (范数) 指标
        self.metrics = {
            'kurtosis': self._compute_kurtosis,
            'entropy': self._compute_entropy,
            'norm': self._compute_norm  # 新增
        }
        self.available_gpus = self._detect_available_gpus()
    
    def _detect_available_gpus(self):
        """Detect available GPUs"""
        if not torch.cuda.is_available():
            return []
        
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                x = torch.randn(100, 100, device=f'cuda:{i}') # 减小测试张量大小加快启动
                del x
                available_gpus.append(i)
                # print(f"GPU {i} is available")
            except Exception as e:
                print(f"GPU {i} is not available: {e}")
        
        return available_gpus
    
    def analyze_model_only_target_modules(self, model, target_modules, target_module_keys, top_k=27):
        """Adaptive analysis using available GPUs"""
        num_modules = len(target_module_keys)
        num_gpus = len(self.available_gpus)

        if num_gpus == 0:
            return self._analyze_cpu(model, target_module_keys, top_k)
        elif num_modules < 20 or num_gpus == 1:
            return self._analyze_single_gpu(model, target_module_keys, top_k)
        else:
            return self._analyze_multi_gpu(model, target_modules, target_module_keys, top_k)
    
    def _group_scores_by_type(self, all_scores, projection_types):
        """将分数按投影类型分组"""
        type_scores = {proj_type: {} for proj_type in projection_types}
        
        def get_projection_type(name):
            for proj_type in projection_types:
                # 兼容 .q_proj. 和 _q_proj 等格式
                if f'{proj_type}' in name:
                    return proj_type
            return None
        
        for name, score in all_scores.items():
            proj_type = get_projection_type(name)
            if proj_type is not None and proj_type in type_scores:
                type_scores[proj_type][name] = score
        
        return type_scores

    # def _compute_final_scores_with_type_aware_norm(self, raw_metrics_dict):
    #     """
    #     核心修改：类型感知归一化 (Type-Aware Normalization)
    #     输入: {'layer_name': {'kurtosis': v, 'entropy': v}}
    #     输出: {'layer_name': final_score}
    #     """
    #     # 1. 按类型收集原始指标
    #     grouped_metrics = {} # {type: {'kurt': [], 'ent': [], 'names': []}}
        
    #     # 自动识别类型
    #     all_types = set()
    #     for name in raw_metrics_dict.keys():
    #         for p_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
    #             if p_type in name:
    #                 all_types.add(p_type)
    #                 break
        
    #     for p_type in all_types:
    #         grouped_metrics[p_type] = {'kurt': [], 'ent': [], 'names': []}

    #     for name, metrics in raw_metrics_dict.items():
    #         for p_type in all_types:
    #             if p_type in name:
    #                 grouped_metrics[p_type]['kurt'].append(metrics['kurtosis'])
    #                 grouped_metrics[p_type]['ent'].append(metrics['entropy'])
    #                 grouped_metrics[p_type]['names'].append(name)
    #                 break
        
    #     final_scores = {}

    #     # 2. 分组归一化并计算分数
    #     for p_type, data in grouped_metrics.items():
    #         if not data['names']: continue
            
    #         kurts = np.array(data['kurt'])
    #         ents = np.array(data['ent'])
            
    #         # Min-Max Normalization (防止除以0)
    #         k_min, k_max = kurts.min(), kurts.max()
    #         e_min, e_max = ents.min(), ents.max()
            
    #         norm_k = (kurts - k_min) / (k_max - k_min + 1e-6)
    #         norm_e = (ents - e_min) / (e_max - e_min + 1e-6)
            
    #         # 新公式：基于 Motivation Proof 结果
    #         # High Kurtosis = Stable (Score up)
    #         # Low Entropy = Stable (Score up) -> (1 - norm_e)
    #         # 权重：Kurtosis 0.6, Entropy 0.4
    #         scores = 0.6 * norm_k + 0.4 * (1 - norm_e)
            
    #         for i, name in enumerate(data['names']):
    #             final_scores[name] = scores[i]
                
    #     return final_scores

    def _compute_final_scores(self, raw_metrics_dict):
        """
        [修改点 4] 分数计算核心逻辑
        目标：分数越高 = 越稳定 = 倾向于共享
        目标：分数越低 = 越不稳定 = 倾向于独享
        """
        grouped = {}
        # 自动识别类型 (q_proj, v_proj...)
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
            
            # Min-Max 归一化函数
            def minmax(arr):
                _min, _max = arr.min(), arr.max()
                if _max - _min < 1e-6: return np.zeros_like(arr) + 0.5
                return (arr - _min) / (_max - _min)

            norm_k = minmax(k)
            norm_e = minmax(e)
            norm_n = minmax(n)
            
            # [公式修改]
            # Norm (权重大小) 与 LoRA更新幅度正相关 -> Norm越大，越不稳定 -> 分数应越低 (1 - norm_n)
            # Kurt/Ent 与 LoRA更新幅度负相关 -> Kurt越大，越稳定 -> 分数应越高 (norm_k)
            # 权重分配: Norm 最重要 (0.4)
            scores = 0.3 * norm_k + 0.3 * norm_e + 0.4 * (1 - norm_n)
            
            for i, name in enumerate(data['names']):
                final_scores[name] = scores[i]
                
        return final_scores
    
    def _select_asymmetric_simple(self, all_scores, base_top_k=20):
        """
        [修改点 5] 简化的非对称分配
        不根据模态计算分数，但根据模态分配名额。
        """
        # 1. 排序：从小到大 (Ascending)
        # Score 低 = 不稳定 = 需要独享 (Exclusive)
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1])
        all_modules = [x[0] for x in sorted_items]
        
        # 2. 简单的非对称逻辑
        # Text 很重要 -> 给 base_top_k 个独享名额
        # AV 不太重要 -> 只给 base_top_k / 2 个独享名额 (只在极度不稳定时独享)
        
        cutoff_text = min(len(all_modules), base_top_k)
        cutoff_av = min(len(all_modules), base_top_k // 2) # [简化逻辑] 简单的减半
        
        text_exclusive_list = all_modules[:cutoff_text]
        av_exclusive_list = all_modules[:cutoff_av]
        
        print(f"\n{'='*40}")
        print(f"SIS Strategy (Simplified Asymmetric)")
        print(f"Text Exclusive Layers (Top-{cutoff_text}): More capacity")
        print(f"AV   Exclusive Layers (Top-{cutoff_av}): Less capacity, more sharing")
        print(f"{'='*40}")

        return {
            "text_exclusive": text_exclusive_list, # 这是一个较大的列表
            "av_exclusive": av_exclusive_list,     # 这是一个较小的列表 (Text的子集)
            "scores": all_scores
        }

    def _analyze_multi_gpu(self, model, projection_types, target_module_keys, top_k):
        # top_k 这里作为一个基准值传入
        """Multi-GPU parallel analysis"""
        print(f"Using multi-GPU parallel analysis with {len(self.available_gpus)} GPUs")
        
        valid_modules = []
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    if weight.numel() >= 100:
                        valid_modules.append((name, weight))
            except Exception:
                continue
        
        if not valid_modules:
            return target_module_keys[:top_k], {}
        
        chunks = self._split_into_chunks(valid_modules, len(self.available_gpus))
        
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(self.available_gpus)) as pool:
            results = []
            for i, chunk in enumerate(chunks):
                if chunk:
                    result = pool.apply_async(
                        self._analyze_chunk_gpu_raw, # 改为调用返回原始指标的函数
                        (chunk, self.available_gpus[i])
                    )
                    results.append(result)
            
            # 收集原始指标
            all_raw_metrics = {}
            for result in results:
                try:
                    chunk_metrics = result.get(timeout=300)
                    all_raw_metrics.update(chunk_metrics)         
                except Exception as e:
                    print(f"Error in GPU process: {e}")

        # # --- 核心修改：在主进程进行类型感知归一化与算分 ---
        # all_scores = self._compute_final_scores_with_type_aware_norm(all_raw_metrics)
        # [修改点 2] 计算分数 (考虑 Norm) #NOTE
        all_scores = self._compute_final_scores(all_raw_metrics)
        
        # 保存分数用于记录
        with open("sis_scores.txt", "w") as f:
            for k, v in sorted(all_scores.items(), key=lambda x: x[1]):
                f.write(f"{k}: {v:.6f}\n")
        
        # # 按类型选择 (使用新的逆序逻辑)
        # # 这里传入 None 作为 type_scores，因为我们在 select 函数里会重新处理
        # return self._select_top_k_global_bottom(all_scores, top_k)

        # [修改点 3] 简化版的非对称分配策略
        return self._select_asymmetric_simple(all_scores, base_top_k=top_k)
    
    def _analyze_chunk_gpu_raw(self, chunk, gpu_id):
        """
        修改版：只计算原始指标，不进行归一化和加权。
        返回: {'name': {'kurtosis': v, 'entropy': v}}
        """
        device = torch.device(f'cuda:{gpu_id}')
        results = {}
        
        for name, weight in chunk:
            try:
                w_gpu = weight.to(device)
                results[name] = {
                    'kurtosis': self._compute_kurtosis_torch(w_gpu),
                    'entropy': self._compute_entropy_torch(w_gpu),
                    'norm': self._compute_norm_torch(w_gpu) # [修改点] 计算 Norm
                }
            except:
                results[name] = {'kurtosis': 0, 'entropy': 0, 'norm': 0}        
        return results
    
    def _analyze_single_gpu(self, model, target_module_keys, top_k):
        """Single GPU analysis"""
        device = torch.device(f'cuda:{self.available_gpus[0]}')
        all_raw_metrics = {}
        
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    if weight.numel() < 100: continue
                    
                    weight_gpu = weight.to(device)
                    kurt = self._compute_kurtosis_torch(weight_gpu)
                    ent = self._compute_entropy_torch(weight_gpu)
                    
                    all_raw_metrics[name] = {'kurtosis': kurt, 'entropy': ent}
            except Exception:
                continue
        
        # 统一归一化算分
        all_scores = self._compute_final_scores_with_type_aware_norm(all_raw_metrics)
        return self._select_top_k_global_bottom(all_scores, top_k)
    
    def _analyze_cpu(self, model, target_module_keys, top_k):
        """CPU fallback"""
        all_raw_metrics = {}
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                weight = module.weight.data
                if weight.numel() < 100: continue
                
                # CPU implementations
                kurt = self._compute_kurtosis(weight)
                ent = self._compute_entropy(weight)
                all_raw_metrics[name] = {'kurtosis': kurt, 'entropy': ent}
            except Exception:
                continue
        
        all_scores = self._compute_final_scores_with_type_aware_norm(all_raw_metrics)
        return self._select_top_k_global_bottom(all_scores, top_k)

    def _select_top_k_global_bottom(self, all_scores, top_k):
        """
        策略修改：逆序选择 (Bottom-K Selection)
        
        逻辑：
        1. Score 越高 = 预训练越稳定 = 微调需求越小 (Shared)
        2. Score 越低 = 预训练越不稳定 = 微调需求越大 (Exclusive)
        3. 目标：选出 top_k 个需要 Exclusive 的层。
        4. 操作：按分数从小到大排序 (Ascending)，取前 k 个。
        """
        
        # 1. 辅助：提取层号
        def extract_layer_num(layer_name):
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try: return int(parts[i + 1])
                    except: continue
            return -1

        # 2. 自动检测 Last Layer (通常也是稳定且不需要大改的，或者作为Safety Net)
        # 根据 Motivation 实验，Last Layer (Output) 的 Norm 其实不高，
        # 所以它应该属于 High Score 区域，自然会被排在后面（Shared）。
        # 但为了稳妥，我们可以不强制保留 Last Layer，完全信任数据。
        # 如果你想强制保留，可以在这里逻辑处理。
        # 这里采用：完全数据驱动 (Data-Driven)。

        # 3. 排序：从小到大 (Reverse=False)
        # 选出分数最低的层 -> 最不稳定的层 -> 最需要 Exclusive LoRA
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=False)
        
        selected_layers = [x[0] for x in sorted_items[:top_k]]
        selected_scores = dict(sorted_items[:top_k])
        
        # 4. 打印统计
        print(f"\n{'='*80}")
        print(f"SIS STRATEGY: BOTTOM-K SELECTION (Low Score = Needs Exclusive LoRA)")
        print(f"{'='*80}")
        print(f"Total Modules Scored: {len(all_scores)}")
        print(f"Selection Target: {top_k}")
        
        # 统计选中的类型
        type_counts = {}
        for name in selected_layers:
            for p_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
                if p_type in name:
                    type_counts[p_type] = type_counts.get(p_type, 0) + 1
                    break
        
        print("\nSelection Breakdown by Type:")
        for p_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {p_type:10s}: {count:3d} modules selected")

        print("\nTop 5 Selected (Lowest Scores - Most Plastic):")
        for i, (name, score) in enumerate(sorted_items[:5]):
            print(f"  {i+1}. {name[-40:]:>40s}: {score:.4f}")
            
        return selected_layers, selected_scores

    def _split_into_chunks(self, data, n_chunks):
        if n_chunks == 0: return [data]
        chunk_size = len(data) // n_chunks + 1
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # --- Metrics Implementations ---
    # CPU
    def _compute_kurtosis(self, tensor):
        if tensor.std() < 1e-8: return 0
        tensor_np = tensor.cpu().numpy().flatten()
        kurt = scipy_kurtosis(tensor_np, fisher=True)
        # 归一化到 0-1 范围，防止极端值
        return 1 / (1 + np.exp(-kurt / 5))
    
    def _compute_entropy(self, tensor, bins=50):
        tensor_np = tensor.cpu().numpy().flatten()
        hist, _ = np.histogram(tensor_np, bins=bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent = scipy_entropy(hist)
        return ent / np.log(bins)
    
    # GPU
    def _compute_kurtosis_torch(self, tensor):
        if tensor.std() < 1e-8: return 0
        mean = tensor.mean()
        std = tensor.std()
        # Fisher Kurtosis = E[(x-u)^4]/std^4 - 3
        kurt = ((tensor - mean) ** 4).mean() / (std ** 4) - 3
        return 1 / (1 + torch.exp(-kurt / 5)).item()

    def _compute_entropy_torch(self, tensor, bins=50):
        min_val, max_val = tensor.min(), tensor.max()
        if (max_val - min_val) < 1e-8: return 0.5
        hist = torch.histc(tensor, bins=bins, min=min_val, max=max_val)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent = -torch.sum(hist * torch.log(hist + 1e-8))
        max_ent = torch.log(torch.tensor(bins, device=tensor.device))
        return (ent / max_ent).item()
    
    def _compute_norm(self, tensor): return torch.norm(tensor.float()).item()
    def _compute_norm_torch(self, tensor): return torch.norm(tensor.float()).item()