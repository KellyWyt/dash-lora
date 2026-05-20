# -*- coding: utf-8 -*-

import jsonlines
import re
import os
import json

# 答案列表（用于参照，宽容模式下主要用于验证）
answer_list = ['zero', 'pipa', 'middle', 'congas', 'eight', 'saxophone', 'tuba', 'no', 'guzheng', 
               'left', 'ten', 'four', 'five', 'nine', 'more than ten', 'drum', 'suona', 'indoor', 
               'two', 'simultaneously', 'piano', 'right', 'acoustic_guitar', 'trumpet', 'seven', 
               'outdoor', 'six', 'yes', 'violin', 'flute', 'clarinet', 'bagpipe', 'one', 'three', 
               'accordion', 'cello', 'electric_bass', 'erhu', 'ukulele', 'bassoon', 'banjo', 
               'xylophone']

def safe_div(numerator, denominator):
    return 100 * numerator / denominator if denominator != 0 else 0.0

def check_lenient(path):
    # 初始化统计字�?
    # 每个类别存两个数：[strict_correct, lenient_correct]
    stats = {
        'A_count': [], 'A_cmp': [],
        'V_count': [], 'V_loc': [],
        'AV_ext': [], 'AV_count': [], 'AV_loc': [], 'AV_cmp': [], 'AV_temp': []
    }
    
    total = 0
    strict_correct_total = 0
    lenient_correct_total = 0

    print(f"Checking: {path}")

    with jsonlines.open(path, 'r') as f:
        for idx, sample in enumerate(f):
            total += 1
            
            # --- 1. 处理 Ground Truth (答案) ---
            answer_raw = sample['output'].split('</s>')[0].strip().lower()
            # 某些答案在列表中�? acoustic_guitar，但文本里可能是 acoustic guitar
            answer_text = answer_raw.replace('_', ' ') 
            
            # --- 2. 处理 Prediction (预测) ---
            pred_raw = sample['predict'].lower()
            # 移除特殊的生成token，保留文�?
            pred_text = pred_raw.replace('<s>', '').replace('</s>', '').replace('<unk>', '')
            
            # --- 3. 严格匹配逻辑 (Strict) ---
            strict_hit = 0
            matches = re.findall(r'<answer>(.*?)</answer>', pred_raw)
            if len(matches) >= 1:
                pred_extracted = matches[0].strip().lower()
                # 只有提取出的内容在答案列表中，且包含答案词，才算严格�?
                if pred_extracted in answer_list:
                    if answer_raw in pred_extracted:
                        strict_hit = 1
            
            # --- 4. 宽容匹配逻辑 (Lenient) ---
            lenient_hit = 0
            # 使用正则 \b 匹配单词边界，防�? "no" 匹配�? "piano" �? "not"
            # re.escape 用于处理可能包含特殊字符的答�?
            pattern = r'\b' + re.escape(answer_text) + r'\b'
            
            if re.search(pattern, pred_text):
                lenient_hit = 1
            
            # 如果答案�? "acoustic_guitar"，我们也尝试匹配带下划线的版�?
            if lenient_hit == 0 and '_' in answer_raw:
                if re.search(r'\b' + re.escape(answer_raw) + r'\b', pred_text):
                    lenient_hit = 1

            # 累加总分
            strict_correct_total += strict_hit
            lenient_correct_total += lenient_hit

            # --- 5. 分类统计 ---
            qt = sample['question_type']
            key = None
            
            if qt[0] == 'Audio':
                if qt[1] == 'Counting': key = 'A_count'
                elif qt[1] == 'Comparative': key = 'A_cmp'
            elif qt[0] == 'Visual':
                if qt[1] == 'Counting': key = 'V_count'
                elif qt[1] == 'Location': key = 'V_loc'
            elif qt[0] == 'Audio-Visual':
                if qt[1] == 'Existential': key = 'AV_ext'
                elif qt[1] == 'Counting': key = 'AV_count'
                elif qt[1] == 'Location': key = 'AV_loc'
                elif qt[1] == 'Comparative': key = 'AV_cmp'
                elif qt[1] == 'Temporal': key = 'AV_temp'
            
            if key:
                # 存入元组 (strict_score, lenient_score)
                stats[key].append((strict_hit, lenient_hit))

    # --- 6. 打印结果表格 ---
    print(f"{'Metric':<25} | {'Strict Acc':<12} | {'Lenient Acc (Hit)':<18} | {'Count':<8}")
    print("-" * 70)
    
    # 辅助函数：计算列表中的平均分
    def calc_scores(key_list):
        if not key_list: return 0.0, 0.0, 0
        n = len(key_list)
        s_sum = sum(x[0] for x in key_list)
        l_sum = sum(x[1] for x in key_list)
        return safe_div(s_sum, n), safe_div(l_sum, n), n

    # 打印各子�?
    for key, val in stats.items():
        s_acc, l_acc, n = calc_scores(val)
        print(f"{key:<25} | {s_acc:>6.2f} %     | {l_acc:>9.2f} %        | {n:<8}")

    print("-" * 70)
    
    # 汇总计�?
    print(f"{'Overall':<25} | {safe_div(strict_correct_total, total):>6.2f} %     | {safe_div(lenient_correct_total, total):>9.2f} %        | {total:<8}")
    
    return total

def main():
    # 替换你的路径
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_replace_A/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/inference_avqa_bs1_tf32_mt150_seed42/results.jsonl'] #原来moka的lora
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_prune_A/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_smart_prune/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_norm_prune/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-7/llama_music1-v3/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed42_test/all_results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-30/checkpoint-2403/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_average/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music1-share-56/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-7/llama_music1-v3/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed42_test/all_results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-3/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-1/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-15/llama_music-share-56-ratio-4/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']

    # results=['/nfs1/WYT/MokA-copy/results/finetune-qwen2/qwen2-1.5B_music-ratio-2/checkpoint-2700/inference_avqa/bs6_tf32_mt150_seed42_test/results.jsonl']




    for result in results:
        if os.path.exists(result):
            check_lenient(result)
        else:
            print(f"File not found: {result}")

if __name__ == "__main__":
    main()