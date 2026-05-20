import json
import os.path as osp
import os
import jsonlines
from alive_progress import alive_it

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def write2json(fp,dict_data,mode='a'):
    with jsonlines.open(fp,mode=mode) as f:
        f.write(dict_data)

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


if __name__ == "__main__":
    results_base = "/nfs1/WYT/MokA/results/finetune/llama_music-mem/checkpoint-1800/inference_avqa/bs6_tf32_mt150_seed42_test_with_msAudio"
    results_dst = osp.join(results_base, "results.jsonl")
    if osp.isfile(results_dst):
        os.remove(results_dst)
    nums = [0,1,2,3]
    results_counts = 0
    results_ids = []

    results_gt = load_json("/nfs1/WYT/MokA/label_file/label_new/music_avqa_test_samples.json")
    results_gt_ids = []
    for ts_data in results_gt:
        ts_id = ts_data["video_id"]
        results_gt_ids.append(ts_id)

    for n_id in nums:
        results_file = osp.join(results_base, f"results{n_id}.jsonl")
        with open(results_file, 'r') as f:
            bar = alive_it(f, finalize=lambda bar: bar.text('Success!'), title=f"Processing {osp.basename(results_file)}")  # <<-- bar with wrapped items
            for line in bar:
            # for line in f:
                json_data = json.loads(line)
                results_id = json_data['vid']
                write2json(fp=results_dst, dict_data=json_data)
                results_counts += 1
                if results_id in results_gt_ids:
                    results_ids.append(results_id)
                else:
                    raise Exception("wrong result!")
    print(f"[CHECK] we need {len(results_gt_ids)} results and got {results_counts} results!")
