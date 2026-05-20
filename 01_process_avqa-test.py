import json
import os.path as osp
import ast

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)
        
"""
[x]: used in class UnifiedTestDataset
json target template:
    {
        "task": "avqa",
        "audio_path": "assets/example/avqa/00006835.mp3",
        "video_path": "assets/example/avqa/00006835.mp4",
        "question": "What is the left instrument of the first sounding instrument?"
    }
json train-valid template:
    {
        "video_id": "00000238",
        "question_id": 8,
        "type": [
            "Audio-Visual",
            "Counting"
        ],
        "video_path": "/group/40061/cserdu/data/music-avqa/video_data/00000238.mp4",
        "audio_path": "/group/40061/cserdu/data/music-avqa/audio_data/00000238.wav",
        "question": "How many instruments are sounding in the video?",
        "answer": "two"
    },
"""

if __name__ == "__main__":
    json_base = "/nfs1/WYT/MokA/label_file/label_new"
    json_src_pth = "avqa-test.json"
    json_dst_pth = "music_avqa_test_samples.json"
    
    json_src = load_json(osp.join(json_base, json_src_pth))
    json_dst = []
    for samp_src in json_src:
        samp_id = samp_src['video_id']
        smap_type = json.loads(samp_src['type'])

        # There are placeholders that have not beed replaced, such as <TH> <LR> etc.
        question = samp_src['question_content'].rstrip().split(' ')
        # question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(samp_src['templ_values'])[p]
                p += 1
            # replace \uff1f with ?
            if '\uff1f' in question[pos]:
                # print(question[pos])
                question[pos] = question[pos][:-1] + '?'

        samp_dst = {
            "video_id": samp_id,
            "question_id": samp_src['question_id'],
            "type": smap_type,
            "video_path": f"/group/40061/cserdu/data/music-avqa/video_data/{samp_id}.mp4",
            "audio_path": f"/group/40061/cserdu/data/music-avqa/audio_data/{samp_id}.wav",
            "question": ' '.join(question),
            "answer": samp_src['anser']
        }
        json_dst.append(samp_dst)
    save_json(json_dst, osp.join(json_base, json_dst_pth), sort_keys=False)
