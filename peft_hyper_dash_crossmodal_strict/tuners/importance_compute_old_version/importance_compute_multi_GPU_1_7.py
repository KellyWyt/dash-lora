# -*- coding: utf-8 -*-

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
            'sparsity': self._compute_sparsity,
            'effective_rank': self._compute_effective_rank
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
                # Test GPU availability
                x = torch.randn(1000, 1000, device=f'cuda:{i}')
                del x
                available_gpus.append(i)
                print(f"GPU {i} is available")
            except Exception as e:
                print(f"GPU {i} is not available: {e}")
        
        return available_gpus
    
    def analyze_model_only_target_modules(self, model, target_modules,target_module_keys, top_k=50):
        """Adaptive analysis using available GPUs"""
        num_modules = len(target_module_keys)
        num_gpus = len(self.available_gpus)

        # print(f"Adaptive analysis: {num_modules} modules, {num_gpus} GPUs available")
        
        if num_gpus == 0:
            # CPU fallback
            return self._analyze_cpu(model, target_module_keys, top_k)
        elif num_modules < 20 or num_gpus == 1:
            # Single GPU for small workloads
            return self._analyze_single_gpu(model, target_module_keys, top_k)
        else:
            # Multi-GPU for large workloads
            return self._analyze_multi_gpu(model, target_modules,target_module_keys, top_k)
    
    def _group_scores_by_type(self, all_scores, projection_types):
        """将分数按投影类型分组"""
        type_scores = {proj_type: {} for proj_type in projection_types}
        
        def get_projection_type(name):
            """从模块名称中提取投影类型"""
            for proj_type in projection_types:
                if f'.{proj_type}' in name or name.endswith(f'.{proj_type}'):
                    return proj_type
            return None
        
        for name, score in all_scores.items():
            proj_type = get_projection_type(name)
            if proj_type is not None and proj_type in type_scores:
                type_scores[proj_type][name] = score
            else:
                # 如果找不到类型，可以记录或忽�?
                pass
        
        return type_scores

    def _print_and_save_results(self, all_scores, all_type_scores, type_statistics):
        """打印和保存结�?"""
        # Print all key-value pairs to console
        print("\n=== All Module Scores ===")
        # for key, value in sorted(all_scores.items()):
        #     print(f"{key}: {value:.6f}")

        # Print type statistics
        print("\n=== Projection Type Statistics ===")
        for proj_type, stats in type_statistics.items():
            print(f"{proj_type:8s}: {stats['count']:3d} layers | "
                f"Avg: {stats['mean']:.4f} | Max: {stats['max']:.4f} | "
                f"Min: {stats['min']:.4f} | Std: {stats['std']:.4f}")

        # Print top layers by type
        print("\n=== Top Layers by Projection Type ===")
        for proj_type, scores_dict in all_type_scores.items():
            if scores_dict:
                # print(f"\n{proj_type} (Top 5):")
                sorted_layers = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                # for i, (name, score) in enumerate(sorted_layers):
                #     # print(f"  {i+1}. {name}: {score:.4f}")

        # Save all scores to file
        with open("all_scores.txt", "w") as f:
            for key, value in sorted(all_scores.items()):
                f.write(f"{key}: {value:.6f}\n")

        # Save type statistics to file
        with open("type_statistics.txt", "w") as f:
            f.write("Projection Type Statistics\n")
            f.write("="*80 + "\n")
            for proj_type, stats in type_statistics.items():
                f.write(f"{proj_type:8s}: {stats['count']:3d} layers | "
                        f"Avg: {stats['mean']:.6f} | Max: {stats['max']:.6f} | "
                        f"Min: {stats['min']:.6f} | Std: {stats['std']:.6f}\n")

        print("\nAll scores saved to all_scores.txt")
        print("Type statistics saved to type_statistics.txt")

    def _analyze_multi_gpu(self, model, projection_types,target_module_keys, top_k):
        """Multi-GPU parallel analysis"""
        print(f"Using multi-GPU parallel analysis with {len(self.available_gpus)} GPUs")
        
        # Collect valid modules
        valid_modules = []
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    if weight.numel() >= 100:
                        valid_modules.append((name, weight))
            except Exception as e:
                print(f"Warning: Could not access module {name}: {e}")
                continue
        
        if not valid_modules:
            return self._fallback_analysis(target_module_keys, top_k)
        
        # Split modules across GPUs
        chunks = self._split_into_chunks(valid_modules, len(self.available_gpus))
        
        # Use multiprocessing for true parallelism
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(self.available_gpus)) as pool:
            results = []
            for i, chunk in enumerate(chunks):
                if chunk:
                    result = pool.apply_async(
                        self._analyze_chunk_gpu,
                        (chunk, self.available_gpus[i])
                    )
                    results.append(result)
            
            # Collect results
            all_scores = {}

            for result in results:
                try:
                    chunk_scores = result.get(timeout=300)  # 5-minute timeout
                    all_scores.update(chunk_scores)         
                except Exception as e:
                    print(f"Error in GPU process: {e}")

        # 现在统一按投影类型分�?
        all_type_scores = self._group_scores_by_type(all_scores, projection_types)
        
        # 计算类型统计
        final_type_statistics = {}
        for proj_type, scores_dict in all_type_scores.items():
            if scores_dict:
                scores_list = list(scores_dict.values())
                final_type_statistics[proj_type] = {
                    'count': len(scores_list),
                    'mean': sum(scores_list) / len(scores_list),
                    'max': max(scores_list),
                    'min': min(scores_list),
                    'median': sorted(scores_list)[len(scores_list)//2],
                    'std': (sum((x - sum(scores_list)/len(scores_list))**2 for x in scores_list)/len(scores_list))**0.5
                }
        
        # 打印和保存结�?
        self._print_and_save_results(all_scores, all_type_scores, final_type_statistics)
        
        # 按类型选择top_k
        return self._select_top_k_by_type(all_scores, all_type_scores, top_k)
    
    def _analyze_chunk_gpu(self, chunk, gpu_id):
        """Analyze a chunk of modules on specific GPU"""
        # projection_types == ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        device = torch.device(f'cuda:{gpu_id}')
        all_scores = {}
        
        for name, weight in chunk:
            try:
                # Move weight to target GPU
                weight_gpu = weight.to(device)
                
                # Calculate scores using PyTorch implementations for speed
                scores = {}
                scores['kurtosis'] = self._compute_kurtosis_torch(weight_gpu)
                scores['entropy'] = self._compute_entropy_torch(weight_gpu)
                scores['sparsity'] = self._compute_sparsity(weight_gpu)
                
                if weight_gpu.dim() >= 2:
                    scores['effective_rank'] = self._compute_effective_rank_torch(weight_gpu)
                else:
                    scores['effective_rank'] = 0.5
                
                # Comprehensive scoring
                final_score = (
                    0.3 * scores['kurtosis'] +
                    0.3 * (1 - scores['entropy']) + 
                    0.2 * scores['sparsity'] +
                    0.2 * scores['effective_rank']
                )

                 # Store scores
                all_scores[name] = final_score
                # print(f"GPU {gpu_id}: Analyzed {name}: score = {final_score:.4f}")
                
            except Exception as e:
                # print(f"GPU {gpu_id}: Error analyzing {name}: {e}")
                all_scores[name] = 0.5
        
        return all_scores
    
    def _analyze_single_gpu(self, model, target_module_keys, top_k):
        """Single GPU analysis"""
        device = torch.device(f'cuda:{self.available_gpus[0]}')
        # print(f"Using single GPU analysis on GPU {self.available_gpus[0]}")
        
        importance_scores = {}
        
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    
                    if weight.numel() < 100:
                        continue
                    
                    # Move to GPU
                    weight_gpu = weight.to(device)
                    
                    # Use PyTorch implementations for speed
                    scores = {}
                    scores['kurtosis'] = self._compute_kurtosis_torch(weight_gpu)
                    scores['entropy'] = self._compute_entropy_torch(weight_gpu)
                    scores['sparsity'] = self._compute_sparsity(weight_gpu)
                    
                    if weight_gpu.dim() >= 2:
                        scores['effective_rank'] = self._compute_effective_rank_torch(weight_gpu)
                    else:
                        scores['effective_rank'] = 0.5
                    
                    importance_scores[name] = (
                        0.3 * scores['kurtosis'] +
                        0.3 * (1 - scores['entropy']) + 
                        0.2 * scores['sparsity'] +
                        0.2 * scores['effective_rank']
                    )
                    # print(f"Single GPU: Analyzed {name}: score = {importance_scores[name]:.4f}")
                    
            except Exception as e:
                print(f"Warning: Could not analyze module {name}: {e}")
                continue
        
        return self._select_top_k(importance_scores, top_k)
    
    def _analyze_cpu(self, model, target_module_keys, top_k):
        """CPU fallback analysis"""
        print("Using CPU analysis")
        
        importance_scores = {}
        
        for name in target_module_keys:
            try:
                module = model.get_submodule(name)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    
                    if weight.numel() < 100:
                        continue
                        
                    scores = {}
                    
                    # Use original implementations for CPU
                    scores['kurtosis'] = self._compute_kurtosis(weight)
                    scores['entropy'] = self._compute_entropy(weight)  
                    scores['sparsity'] = self._compute_sparsity(weight)
                    
                    if weight.dim() >= 2:
                        scores['effective_rank'] = self._compute_effective_rank(weight)
                    else:
                        scores['effective_rank'] = 0.5
                    
                    importance_scores[name] = (
                        0.3 * scores['kurtosis'] +
                        0.3 * (1 - scores['entropy']) + 
                        0.2 * scores['sparsity'] +
                        0.2 * scores['effective_rank']
                    )
                    # print(f"CPU: Analyzed {name}: score = {importance_scores[name]:.4f}")
                    
            except Exception as e:
                print(f"Warning: Could not analyze module {name}: {e}")
                continue
        
        return self._select_top_k(importance_scores, top_k)
    
    def _split_into_chunks(self, data, n_chunks):
        """Split data into n chunks"""
        if n_chunks == 0:
            return [data]
            
        chunk_size = len(data) // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            if i == n_chunks - 1:
                end = len(data)
            else:
                end = start + chunk_size
            chunks.append(data[start:end])
        return chunks
    
    def _fallback_analysis(self, target_module_keys, top_k):
        """Fallback when no valid modules found"""
        print("Warning: No valid modules could be analyzed, using fallback")
        return target_module_keys[:top_k], {name: 1.0 for name in target_module_keys}
    
    def _select_top_k_by_type1(self, all_scores, type_scores, top_k):
        top_k_layers = []
        final_scores = {}
        
        # 计算每个类型应该分配的名�?
        total_layers = len(all_scores)
        type_quotas = {}
        
        for proj_type, scores_dict in type_scores.items():
            if scores_dict:
                # 按比例分配名�?
                proportion = len(scores_dict) / total_layers
                k_for_type = max(1, int(top_k * proportion))
                type_quotas[proj_type] = min(k_for_type, len(scores_dict))
        
        # 确保总和不超过top_k
        while sum(type_quotas.values()) > top_k:
            # 减少名额最多的类型
            max_type = max(type_quotas.items(), key=lambda x: x[1])[0]
            type_quotas[max_type] -= 1
        
        # 为每个类型选择前k�?
        for proj_type, k_for_type in type_quotas.items():
            if proj_type in type_scores and type_scores[proj_type]:
                sorted_type = sorted(type_scores[proj_type].items(), 
                                key=lambda x: x[1], reverse=True)[:k_for_type]
                for layer_name, score in sorted_type:
                    top_k_layers.append(layer_name)
                    final_scores[layer_name] = score

        return top_k_layers,final_scores
        
    def _select_top_k_by_type(self, all_scores, type_scores, top_k):
        #TODO '''statistic allocate'''
        """选择top_k模块，排�?31层，但最终返回包�?31层所有模�?"""
        # 要排除的层号
        exclude_layer_num = 31
        exclude_patterns = [
            f"layers.{exclude_layer_num}.",
            f".layers.{exclude_layer_num}.",
            f"layers.{exclude_layer_num}.", 
            f".{exclude_layer_num}."
        ]
        
        # 辅助函数：提取层�?
        def extract_layer_num(layer_name):
            """从层名中提取层号"""
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        continue
            return -1  # 无法提取
        
        # 分离31层和�?31�?
        layer_31_scores = {}  # 31层的分数
        non_31_scores = {}    # �?31层的分数
        non_31_type_scores = {}  # �?31层按类型分组的分�?
        
        for name, score in all_scores.items():
            layer_num = extract_layer_num(name)
            if layer_num == exclude_layer_num:
                layer_31_scores[name] = score
            else:
                non_31_scores[name] = score
        
        # 统计31层各类型的数�?
        layer_31_type_counts = {}
        for name in layer_31_scores.keys():
            # 提取类型
            if 'q_proj' in name:
                proj_type = 'q_proj'
            elif 'k_proj' in name:
                proj_type = 'k_proj'
            elif 'v_proj' in name:
                proj_type = 'v_proj'
            elif 'o_proj' in name:
                proj_type = 'o_proj'
            elif 'gate_proj' in name:
                proj_type = 'gate_proj'
            elif 'down_proj' in name:
                proj_type = 'down_proj'
            elif 'up_proj' in name:
                proj_type = 'up_proj'
            else:
                continue
                
            layer_31_type_counts[proj_type] = layer_31_type_counts.get(proj_type, 0) + 1
        
        # 重新组织�?31层按类型分组的分�?
        for proj_type, scores_dict in type_scores.items():
            if scores_dict:
                non_31_type_scores[proj_type] = {}
                for name, score in scores_dict.items():
                    layer_num = extract_layer_num(name)
                    if layer_num != exclude_layer_num:
                        non_31_type_scores[proj_type][name] = score

        top_k_layers = []
        final_scores = {}
        
        # 静态配额分配（基于您的分数统计分析�?
        if top_k == 49:
            # top_k=50时的推荐配额
            recommended_quotas = {
                'o_proj': 11,      # 22% - 最高平均分
                'down_proj': 10,   # 20% - 高平均分且稳�?
                'q_proj': 9,       # 18% - 高平均分
                'gate_proj': 6,    # 12% - 中等分数
                'up_proj': 5,      # 10% - 中等分数
                'k_proj': 5,       # 10% - 较低分数
                'v_proj': 4,       # 8%  - 最低平均分
            }
        elif top_k == 30:
            # top_k=30时的推荐配额（按比例缩小�?
            recommended_quotas = {
                'o_proj': 7,       # 23.3%
                'down_proj': 6,    # 20.0%
                'q_proj': 5,       # 16.7%
                'gate_proj': 4,    # 13.3%
                'up_proj': 3,      # 10.0%
                'k_proj': 3,       # 10.0%
                'v_proj': 2,       # 6.7%
            }
        elif top_k == 80:
            # top_k=80时的推荐配额（按比例扩大�?
            recommended_quotas = {
                'o_proj': 17,      # 21.3%
                'down_proj': 16,   # 20.0%
                'q_proj': 14,      # 17.5%
                'gate_proj': 10,   # 12.5%
                'up_proj': 8,      # 10.0%
                'k_proj': 8,       # 10.0%
                'v_proj': 7,       # 8.7%
            }
        else:
            # 对于其他top_k值，使用动态比例计�?
            scale_factor = top_k / 49.0  # �?50为基�?
            recommended_quotas = {
                'o_proj': max(1, int(11 * scale_factor)),
                'down_proj': max(1, int(10 * scale_factor)),
                'q_proj': max(1, int(9 * scale_factor)),
                'gate_proj': max(1, int(6 * scale_factor)),
                'up_proj': max(1, int(5 * scale_factor)),
                'k_proj': max(1, int(5 * scale_factor)),
                'v_proj': max(1, int(4 * scale_factor)),
            }
        
        # 确保配额不超过每个类型的实际层数
        adjusted_quotas = {}
        for proj_type, quota in recommended_quotas.items():
            if proj_type in type_scores and type_scores[proj_type]:
                max_possible = len(type_scores[proj_type])
                adjusted_quotas[proj_type] = min(quota, max_possible)
            else:
                adjusted_quotas[proj_type] = 0
        
        # 调整配额总和为top_k
        current_total = sum(adjusted_quotas.values())
        
        if current_total > top_k:
            # 需要减少配额：从配额最多的类型开始减
            while current_total > top_k:
                max_type = max(adjusted_quotas.items(), key=lambda x: x[1])[0]
                if adjusted_quotas[max_type] > 1:  # 至少保留1�?
                    adjusted_quotas[max_type] -= 1
                    current_total -= 1
                else:
                    # 如果某个类型只有1个配额，找下一�?
                    break
        
        elif current_total < top_k:
            # 需要增加配额：从配额最少的类型开始加
            while current_total < top_k:
                # 优先增加高分类型（o_proj, down_proj, q_proj�?
                priority_types = ['o_proj', 'down_proj', 'q_proj', 'gate_proj', 'up_proj', 'k_proj', 'v_proj']
                for proj_type in priority_types:
                    if proj_type in adjusted_quotas:
                        max_possible = len(non_31_type_scores.get(proj_type, {}))
                        if adjusted_quotas[proj_type] < max_possible:
                            adjusted_quotas[proj_type] += 1
                            current_total += 1
                            break
        
        # 为每个类型从�?31层中选择前k�?
        selected_non_31_layers = []
        selected_non_31_scores = {}

        # 为每个类型选择前k�?
        for proj_type, k_for_type in adjusted_quotas.items():
            if proj_type in non_31_type_scores and non_31_type_scores[proj_type] and k_for_type > 0:
                sorted_type = sorted(non_31_type_scores[proj_type].items(), 
                                key=lambda x: x[1], reverse=True)[:k_for_type]
                for layer_name, score in sorted_type:
                    selected_non_31_layers.append(layer_name)
                    selected_non_31_scores[layer_name] = score
        
        # 合并结果�?31层所有模�? + �?31层选中的top_k模块
        top_k_layers = list(layer_31_scores.keys()) + selected_non_31_layers
        final_scores = {**layer_31_scores, **selected_non_31_scores}
        
        # 打印分配详情
        print(f"\n{'='*80}")
        print(f"STATIC QUOTA ALLOCATION (top_k={top_k}, excluding layer {exclude_layer_num})")
        print(f"{'='*80}")
        
        # 打印31层模块统�?
        print(f"\nLayer {exclude_layer_num} modules (automatically included):")
        print(f"Total: {len(layer_31_scores)} modules")
        print("Breakdown by type:")
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
            count = layer_31_type_counts.get(proj_type, 0)
            if count > 0:
                print(f"  {proj_type:10s}: {count} module(s)")
        
        # 计算每个类型的实际平均分（仅�?31层）
        type_avg_scores = {}
        type_layer_counts = {}
        for proj_type, scores_dict in non_31_type_scores.items():
            if scores_dict:
                type_avg_scores[proj_type] = sum(scores_dict.values()) / len(scores_dict)
                type_layer_counts[proj_type] = len(scores_dict)
        
        # 按平均分排序打印
        sorted_types = sorted(type_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} selection from non-{exclude_layer_num} layers:")
        print(f"{'Type':10s} {'Avg Score':10s} {'Available':10s} {'Quota':6s} {'% of Total':10s} {'Selected Layers'}")
        print(f"{'-'*80}")
        
        total_selected_non_31 = sum(adjusted_quotas.values())
        for proj_type, avg_score in sorted_types:
            quota = adjusted_quotas.get(proj_type, 0)
            available = type_layer_counts.get(proj_type, 0)
            percentage = (quota / total_selected_non_31 * 100) if total_selected_non_31 > 0 else 0
            
            # 获取选中的层名（简化显示）
            if proj_type in non_31_type_scores and quota > 0:
                selected_layers = sorted(non_31_type_scores[proj_type].items(), 
                                    key=lambda x: x[1], reverse=True)[:quota]
                # 提取层号
                layer_nums = []
                for name, _ in selected_layers:
                    layer_num = extract_layer_num(name)
                    if layer_num != -1:
                        layer_nums.append(layer_num)
                
                # 排序并去�?
                layer_nums = sorted(set(layer_nums))
                if layer_nums:
                    layers_str = f"{len(selected_layers)} layers from: " + ", ".join([f"layer_{num}" for num in layer_nums[:5]])
                    if len(layer_nums) > 5:
                        layers_str += f" ... (+{len(layer_nums)-5} more layers)"
                else:
                    layers_str = f"{len(selected_layers)} layers"
            else:
                layers_str = "None"
            
            print(f"{proj_type:10s} {avg_score:9.4f}  {available:10d}  {quota:6d}  {percentage:8.1f}%  {layers_str}")
        
        print(f"{'='*80}")
        print(f"Summary:")
        print(f"  - Layer {exclude_layer_num} modules: {len(layer_31_scores)} layers (automatically included)")
        print(f"  - Selected from non-{exclude_layer_num} layers: {total_selected_non_31} layers")
        print(f"  - Total selected: {len(top_k_layers)} layers")
        print(f"{'='*80}")
        
        # 打印详细的层信息
        print(f"\nDETAILED MODULE INFORMATION:")
        
        # 打印31层模�?
        if layer_31_scores:
            print(f"\n1. Layer {exclude_layer_num} modules ({len(layer_31_scores)} modules):")
            # 按类型分组显�?
            layer_31_by_type = {}
            for name, score in layer_31_scores.items():
                # 提取类型
                for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
                    if proj_type in name:
                        if proj_type not in layer_31_by_type:
                            layer_31_by_type[proj_type] = []
                        layer_31_by_type[proj_type].append((name, score))
                        break
            
            # 按类型打�?
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
                if proj_type in layer_31_by_type:
                    modules = layer_31_by_type[proj_type]
                    print(f"   {proj_type}:")
                    for name, score in sorted(modules, key=lambda x: x[1], reverse=True):
                        # 简化显示名�?
                        parts = name.split('.')
                        simple_name = '.'.join(parts[-2:]) if len(parts) >= 2 else name
                        print(f"     - {simple_name:30s} : {score:.4f}")
        
        # 打印�?31层选中的模�?
        if selected_non_31_scores:
            print(f"\n2. Top {top_k} modules from non-{exclude_layer_num} layers ({len(selected_non_31_scores)} modules):")
            # 按分数排�?
            sorted_non_31 = sorted(selected_non_31_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 按类型分组统�?
            non_31_by_type = {}
            for name, score in sorted_non_31:
                # 提取类型
                for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']:
                    if proj_type in name:
                        if proj_type not in non_31_by_type:
                            non_31_by_type[proj_type] = []
                        non_31_by_type[proj_type].append((name, score))
                        break
            
            # 按类型打印统�?
            print("   Breakdown by type:")
            for proj_type in ['o_proj', 'down_proj', 'q_proj', 'gate_proj', 'up_proj', 'k_proj', 'v_proj']:
                if proj_type in non_31_by_type:
                    count = len(non_31_by_type[proj_type])
                    avg_score = sum(score for _, score in non_31_by_type[proj_type]) / count
                    print(f"     {proj_type:10s}: {count:2d} modules, avg score: {avg_score:.4f}")
            
            # 打印详细列表
            print(f"\n   Detailed list (sorted by score):")
            for i, (layer_name, score) in enumerate(sorted_non_31):
                # 提取简化的层名
                parts = layer_name.split('.')
                if len(parts) >= 4:
                    # 提取层号
                    layer_num = "unknown"
                    for j, part in enumerate(parts):
                        if part == 'layers' and j + 1 < len(parts):
                            layer_num = parts[j + 1]
                            break
                    simple_name = f"layer_{layer_num}.{parts[-2]}.{parts[-1]}"
                else:
                    simple_name = layer_name
                
                # 打印�?20个或所�?
                if i < 20 or len(sorted_non_31) <= 30:
                    print(f"     {i+1:3d}. {simple_name:50s} : {score:.4f}")
                elif i == 20:
                    print(f"     ... (showing top 20 of {len(sorted_non_31)})")
                    break
    
        return top_k_layers, final_scores

    def _select_top_k_by_type_no_except_31(self, all_scores, type_scores, top_k):
        #TODO '''statistic allocate'''
        top_k_layers = []
        final_scores = {}
        
        # 静态配额分配（基于您的分数统计分析�?
        if top_k == 50:
            # top_k=50时的推荐配额
            recommended_quotas = {
                'o_proj': 11,      # 22% - 最高平均分
                'down_proj': 10,   # 20% - 高平均分且稳�?
                'q_proj': 9,       # 18% - 高平均分
                'gate_proj': 6,    # 12% - 中等分数
                'up_proj': 5,      # 10% - 中等分数
                'k_proj': 5,       # 10% - 较低分数
                'v_proj': 4,       # 8%  - 最低平均分
            }
        elif top_k == 30:
            # top_k=30时的推荐配额（按比例缩小�?
            recommended_quotas = {
                'o_proj': 7,       # 23.3%
                'down_proj': 6,    # 20.0%
                'q_proj': 5,       # 16.7%
                'gate_proj': 4,    # 13.3%
                'up_proj': 3,      # 10.0%
                'k_proj': 3,       # 10.0%
                'v_proj': 2,       # 6.7%
            }
        elif top_k == 80:
            # top_k=80时的推荐配额（按比例扩大�?
            recommended_quotas = {
                'o_proj': 17,      # 21.3%
                'down_proj': 16,   # 20.0%
                'q_proj': 14,      # 17.5%
                'gate_proj': 10,   # 12.5%
                'up_proj': 8,      # 10.0%
                'k_proj': 8,       # 10.0%
                'v_proj': 7,       # 8.7%
            }
        else:
            # 对于其他top_k值，使用动态比例计�?
            scale_factor = top_k / 50.0  # �?50为基�?
            recommended_quotas = {
                'o_proj': max(1, int(11 * scale_factor)),
                'down_proj': max(1, int(10 * scale_factor)),
                'q_proj': max(1, int(9 * scale_factor)),
                'gate_proj': max(1, int(6 * scale_factor)),
                'up_proj': max(1, int(5 * scale_factor)),
                'k_proj': max(1, int(5 * scale_factor)),
                'v_proj': max(1, int(4 * scale_factor)),
            }
        
        # 确保配额不超过每个类型的实际层数
        adjusted_quotas = {}
        for proj_type, quota in recommended_quotas.items():
            if proj_type in type_scores and type_scores[proj_type]:
                max_possible = len(type_scores[proj_type])
                adjusted_quotas[proj_type] = min(quota, max_possible)
            else:
                adjusted_quotas[proj_type] = 0
        
        # 调整配额总和为top_k
        current_total = sum(adjusted_quotas.values())
        
        if current_total > top_k:
            # 需要减少配额：从配额最多的类型开始减
            while current_total > top_k:
                max_type = max(adjusted_quotas.items(), key=lambda x: x[1])[0]
                if adjusted_quotas[max_type] > 1:  # 至少保留1�?
                    adjusted_quotas[max_type] -= 1
                    current_total -= 1
                else:
                    # 如果某个类型只有1个配额，找下一�?
                    break
        
        elif current_total < top_k:
            # 需要增加配额：从配额最少的类型开始加
            while current_total < top_k:
                # 优先增加高分类型（o_proj, down_proj, q_proj�?
                priority_types = ['o_proj', 'down_proj', 'q_proj', 'gate_proj', 'up_proj', 'k_proj', 'v_proj']
                for proj_type in priority_types:
                    if proj_type in adjusted_quotas:
                        max_possible = len(type_scores.get(proj_type, {}))
                        if adjusted_quotas[proj_type] < max_possible:
                            adjusted_quotas[proj_type] += 1
                            current_total += 1
                            break
        
        # 为每个类型选择前k�?
        for proj_type, k_for_type in adjusted_quotas.items():
            if proj_type in type_scores and type_scores[proj_type] and k_for_type > 0:
                sorted_type = sorted(type_scores[proj_type].items(), 
                                key=lambda x: x[1], reverse=True)[:k_for_type]
                for layer_name, score in sorted_type:
                    top_k_layers.append(layer_name)
                    final_scores[layer_name] = score
        
        # 打印分配详情
        print(f"\n{'='*80}")
        print(f"STATIC QUOTA ALLOCATION (top_k={top_k})")
        print(f"{'='*80}")
        
        # 计算每个类型的实际平均分
        type_avg_scores = {}
        for proj_type, scores_dict in type_scores.items():
            if scores_dict:
                type_avg_scores[proj_type] = sum(scores_dict.values()) / len(scores_dict)
        
        # 按平均分排序打印
        sorted_types = sorted(type_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Type':10s} {'Avg Score':10s} {'Quota':6s} {'% of Total':10s} {'Selected Layers'}")
        print(f"{'-'*80}")
        
        total_selected = sum(adjusted_quotas.values())
        for proj_type, avg_score in sorted_types:
            quota = adjusted_quotas.get(proj_type, 0)
            percentage = (quota / total_selected * 100) if total_selected > 0 else 0
            
            # 获取选中的层名（简化显示）
            if proj_type in type_scores and quota > 0:
                selected_layers = sorted(type_scores[proj_type].items(), 
                                    key=lambda x: x[1], reverse=True)[:quota]
                layer_names = [name.split('.')[-3] if '.' in name else name for name, _ in selected_layers]
                layers_str = f"{len(selected_layers)} layers: " + ", ".join(layer_names[:3])
                if len(selected_layers) > 3:
                    layers_str += f" ... (+{len(selected_layers)-3} more)"
            else:
                layers_str = "None"
            
            print(f"{proj_type:10s} {avg_score:9.4f}  {quota:6d}  {percentage:8.1f}%  {layers_str}")
        
        print(f"{'='*80}")
        print(f"Total selected: {total_selected} layers (target: {top_k})")
        print(f"{'='*80}")
        
        # 打印选中的层详情
        # print(f"\nSELECTED LAYERS (sorted by score):")
        sorted_selected = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (layer_name, score) in enumerate(sorted_selected):
            # 提取简化的层名
            parts = layer_name.split('.')
            if len(parts) >= 3:
                simple_name = f"{parts[-3]}.{parts[-2]}.{parts[-1]}"
            else:
                simple_name = layer_name
            
            # print(f"{i+1:3d}. {simple_name:50s} : {score:.4f}")
        
        return top_k_layers, final_scores


    def _select_top_k(self, importance_scores, top_k):
        """Select top-k important layers"""
        if importance_scores:
            sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_layers = [layer_name for layer_name, score in sorted_layers[:top_k]]
            
            print(f"Successfully analyzed {len(importance_scores)} target modules")
            return top_k_layers, importance_scores
        else:
            print("Warning: No target modules could be analyzed")
            return [], {}
    
    # Original metric implementations (for CPU)
    def _compute_kurtosis(self, tensor):
        if tensor.std() < 1e-8:
            return 0
        tensor_np = tensor.cpu().numpy().flatten()
        kurt = scipy_kurtosis(tensor_np, fisher=True)
        return 1 / (1 + np.exp(-kurt / 5))
    
    def _compute_entropy(self, tensor, bins=50):
        tensor_np = tensor.cpu().numpy().flatten()
        hist, _ = np.histogram(tensor_np, bins=bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent = scipy_entropy(hist)
        return ent / np.log(bins)
    
    # PyTorch implementations (for GPU)
    def _compute_kurtosis_torch(self, tensor):
        if tensor.std() < 1e-8:
            return 0
        mean = tensor.mean()
        std = tensor.std()
        fourth_moment = ((tensor - mean) ** 4).mean()
        kurt = fourth_moment / (std ** 4) - 3
        return 1 / (1 + torch.exp(-kurt / 5)).item()

    def _compute_entropy_torch(self, tensor, bins=50):
        min_val = tensor.min()
        max_val = tensor.max()
        if (max_val - min_val) < 1e-8:
            return 0.5
        
        hist = torch.histc(tensor, bins=bins, min=min_val, max=max_val)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent = -torch.sum(hist * torch.log(hist + 1e-8))  # Avoid log(0)
        max_ent = torch.log(torch.tensor(bins, device=tensor.device))
        return (ent / max_ent).item()
    
    def _compute_effective_rank_torch(self, tensor, threshold=0.01):
        if tensor.dim() < 2:
            return 0.5
        try:
            # Use randomized SVD for large matrices
            if tensor.shape[0] > 1000 or tensor.shape[1] > 1000:
                U, S, V = torch.svd_lowrank(tensor.float(), q=100)
            else:
                U, S, V = torch.svd(tensor.float())
            S = S / S.max()
            effective_rank = (S > threshold).sum().float() / len(S)
            return effective_rank.item()
        except:
            return 0.5

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