import jsonlines
import re
import os
import json

answer_list = ['zero', 'pipa', 'middle', 'congas', 'eight', 'saxophone', 'tuba', 'no', 'guzheng', 
               'left', 'ten', 'four', 'five', 'nine', 'more than ten', 'drum', 'suona', 'indoor', 
               'two', 'simultaneously', 'piano', 'right', 'acoustic_guitar', 'trumpet', 'seven', 
               'outdoor', 'six', 'yes', 'violin', 'flute', 'clarinet', 'bagpipe', 'one', 'three', 
               'accordion', 'cello', 'electric_bass', 'erhu', 'ukulele', 'bassoon', 'banjo', 
               'xylophone']

def safe_div(numerator, denominator):
    """安全除法，防止分母为0"""
    if denominator == 0:
        return 0.0
    return 100 * numerator / denominator

def check(path):
    # 初始化统计列表
    stats = {
        'A_count': [], 'A_cmp': [],
        'V_count': [], 'V_loc': [],
        'AV_ext': [], 'AV_count': [], 'AV_loc': [], 'AV_cmp': [], 'AV_temp': []
    }

    correct = 0
    total = 0
    real_count = 0

    with jsonlines.open(path, 'r') as f:
        for idx, sample in enumerate(f):
            real_count += 1
            answer = sample['output']
            # 清理答案格式
            answer = answer.split('</s>')[0].strip().lower()
            
            raw_pred = sample['predict']
            question_type = sample['question_type']
            
            # 尝试提取答案
            matches = re.findall(r'<answer>(.*?)</answer>', raw_pred)
            
            pred = ""
            if len(matches) >= 1:
                # 如果有匹配，取第一个（或者你可以根据逻辑取最后一个）
                pred = matches[0].strip().lower()
            else:
                # 【修改点1】如果提取失败，不continue，而是记录为空字符串，算作错误
                # print(f'[FORMAT ERROR] row: {idx+1} No <answer> tag found. Pred: {raw_pred[:50]}...')
                pred = "" 

            # 判断正误
            pred_true = 0
            # 只有当提取到了内容，且内容在答案列表中，且答案包含在预测中时才算对
            if pred and (pred in answer_list):
                if answer in pred: # 这里逻辑是只要 answer 是 pred 的子串就算对，比如 answer="yes", pred="yes"
                     pred_true = 1
            
            # 记录错误日志（可选）
            # if pred_true == 0:
            #     print(f'[WRONG] row: {idx+1} answer: {answer} pred_extracted: {pred}')

            total += 1
            correct += pred_true

            # 【修改点2】分类统计逻辑保持不变，但确保无论是否提取到tag，都会append结果(0或1)
            q_main = question_type[0]
            q_sub = question_type[1]

            if q_main == 'Audio':
                if q_sub == 'Counting': stats['A_count'].append(pred_true)
                elif q_sub == 'Comparative': stats['A_cmp'].append(pred_true)
            elif q_main == 'Visual':
                if q_sub == 'Counting': stats['V_count'].append(pred_true)
                elif q_sub == 'Location': stats['V_loc'].append(pred_true)
            elif q_main == 'Audio-Visual':
                if q_sub == 'Existential': stats['AV_ext'].append(pred_true)
                elif q_sub == 'Counting': stats['AV_count'].append(pred_true)
                elif q_sub == 'Location': stats['AV_loc'].append(pred_true)
                elif q_sub == 'Comparative': stats['AV_cmp'].append(pred_true)
                elif q_sub == 'Temporal': stats['AV_temp'].append(pred_true)

    # 【修改点3】使用安全除法输出结果，防止报错
    print('-' * 30)
    print(f"Total processed: {total}")
    
    print('Audio Counting Accuracy: %.2f %% (%d/%d)' % (
            safe_div(sum(stats['A_count']), len(stats['A_count'])), sum(stats['A_count']), len(stats['A_count'])))
    
    print('Audio Cmp Accuracy:      %.2f %% (%d/%d)' % (
            safe_div(sum(stats['A_cmp']), len(stats['A_cmp'])), sum(stats['A_cmp']), len(stats['A_cmp'])))
    
    # Audio Total
    a_total_hits = sum(stats['A_count']) + sum(stats['A_cmp'])
    a_total_len = len(stats['A_count']) + len(stats['A_cmp'])
    print('Audio Accuracy:          %.2f %%' % safe_div(a_total_hits, a_total_len))

    print('Visual Counting Accuracy: %.2f %% (%d/%d)' % (
            safe_div(sum(stats['V_count']), len(stats['V_count'])), sum(stats['V_count']), len(stats['V_count'])))
    
    print('Visual Loc Accuracy:      %.2f %% (%d/%d)' % (
            safe_div(sum(stats['V_loc']), len(stats['V_loc'])), sum(stats['V_loc']), len(stats['V_loc'])))
    
    # Visual Total
    v_total_hits = sum(stats['V_count']) + sum(stats['V_loc'])
    v_total_len = len(stats['V_count']) + len(stats['V_loc'])
    print('Visual Accuracy:          %.2f %%' % safe_div(v_total_hits, v_total_len))

    print('AV Ext Accuracy:          %.2f %% (%d/%d)' % (
            safe_div(sum(stats['AV_ext']), len(stats['AV_ext'])), sum(stats['AV_ext']), len(stats['AV_ext'])))
    print('AV counting Accuracy:     %.2f %% (%d/%d)' % (
            safe_div(sum(stats['AV_count']), len(stats['AV_count'])), sum(stats['AV_count']), len(stats['AV_count'])))
    print('AV Loc Accuracy:          %.2f %% (%d/%d)' % (
            safe_div(sum(stats['AV_loc']), len(stats['AV_loc'])), sum(stats['AV_loc']), len(stats['AV_loc'])))
    print('AV Cmp Accuracy:          %.2f %% (%d/%d)' % (
            safe_div(sum(stats['AV_cmp']), len(stats['AV_cmp'])), sum(stats['AV_cmp']), len(stats['AV_cmp'])))
    print('AV Temporal Accuracy:     %.2f %% (%d/%d)' % (
            safe_div(sum(stats['AV_temp']), len(stats['AV_temp'])), sum(stats['AV_temp']), len(stats['AV_temp'])))

    # AV Total
    av_total_hits = sum(stats['AV_count']) + sum(stats['AV_loc']) + sum(stats['AV_ext']) + sum(stats['AV_temp']) + sum(stats['AV_cmp'])
    av_total_len = len(stats['AV_count']) + len(stats['AV_loc']) + len(stats['AV_ext']) + len(stats['AV_temp']) + len(stats['AV_cmp'])
    print('AV Accuracy:              %.2f %%' % safe_div(av_total_hits, av_total_len))

    print('-' * 30)
    print('Overall Accuracy: %.2f %%' % safe_div(correct, total))
    print(f'correct: {correct} total: {total}')
    print('[Rectified] Overall Accuracy: %.2f %%' % safe_div(correct, real_count))
    print(f'[Rectified] correct: {correct} total: {real_count}')

    return total, safe_div(correct, total)

def main():
    # 替换为你实际的文件路径
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_motivation/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_prune_A/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/inference_avqa_bs1_tf32_mt150_seed42/results.jsonl']

    for result in results:
        print(f'Processing: {result}')
        if os.path.exists(result):
            num, acc = check(result)
            print('nums: %d acc:  %.2f %%' % (num, acc))
        else:
            print(f"File not found: {result}")

if __name__ == "__main__":
    main()