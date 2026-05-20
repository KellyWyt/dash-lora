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






def check(path):

    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    correct = 0
    total = 0
    real_count = 0

    with jsonlines.open(path,'r') as f:
        for idx, sample in enumerate(f):
            real_count += 1
            answer = sample['output']
            # answer = answer.split('</s>')[0] <|endoftext|>
            answer = answer.split('<|endoftext|>')[0] 

            pred = sample['predict']
            question_type = sample['question_type']
            matches = re.findall(r'<answer>(.*?)</answer>', pred)
            if len(matches) != 1:
                print(f'[BUG] row: {idx+1} answer: {answer} pred: {pred}')
                continue
            pred = matches[0].strip()
            answer = answer.strip().lower()
            pred = pred.strip().lower()
            pred_true = 0
            if pred not in answer_list:
                print(f'[WRONG] row: {idx+1} answer: {answer} pred: {pred}')
                # continue
            else:
                pred = pred.lower()
                if answer in pred:
                    pred_true = 1
            
            total += 1
            correct += pred_true

            if question_type[0] == 'Audio':
                if question_type[1] == 'Counting':
                    A_count.append(pred_true)
                elif question_type[1] == 'Comparative':
                    A_cmp.append(pred_true)
            elif question_type[0] == 'Visual':
                if question_type[1] == 'Counting':
                    V_count.append(pred_true)
                elif question_type[1] == 'Location':
                    V_loc.append(pred_true)
            elif question_type[0] == 'Audio-Visual':
                if question_type[1] == 'Existential':
                    AV_ext.append(pred_true)
                elif question_type[1] == 'Counting':
                    AV_count.append(pred_true)
                elif question_type[1] == 'Location':
                    AV_loc.append(pred_true)
                elif question_type[1] == 'Comparative':
                    AV_cmp.append(pred_true)
                elif question_type[1] == 'Temporal':
                    AV_temp.append(pred_true)

    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                    +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    print(f'correct: {correct} total: {total}')
    print('[Reftified] Overall Accuracy: %.2f %%' % (
            100 * correct / real_count))
    print(f'[Reftified] correct: {correct} total: {real_count}')


    return total,100 * correct / total



def main():
    # sed42 - 77.89 | sed123 - 76.54 | sed456: 77.34 -> 77.26
    # results=['music_avqa/checkpoint-675/inference_avqa/results_0.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune-share-1-7/llama_music1-v3/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed42_test/all_results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800/inference_avqa_bs1_tf32_mt150_seed42/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_motivation/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/results/finetune/llama_music/checkpoint-1800_test_prune_A/inference_avqa/bs6_tf32_mt150_seed123_test/results.jsonl']
#     results=['/nfs1/WYT/MokA/results/finetune/llama_music-mem/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed42_test_with_msAudio/results.jsonl']
    # results=['/nfs1/WYT/MokA/results/finetune/llama_music-mem2/checkpoint-1800/inference_avqa_bs6_tf32_mt150_seed42/results.jsonl']
    # results=['/nfs1/WYT/MokA-copy/changeForQwen2/inferenceFromMoka+Qwen2-1.5B-Instruct/inference_avqa.jsonl']
    results=['/nfs1/WYT/MokA-copy/results/finetune-qwen2/qwen2-1.5B_music-ratio-2/checkpoint-2700/inference_avqa/bs6_tf32_mt150_seed42_test/results.jsonl']



    for result in results:
        num,acc=check(result)
        print('result: %s'%result)
        print(' nums: %d acc:  %.2f %%'% (num,acc))




if __name__ == "__main__":
    main()
