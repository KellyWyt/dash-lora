import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi

class AudioProcessor:

    def __init__(
        self,
        sr = 16000,
        mono = True,
        duration = 60,
    ) -> None:
        self.sr = sr
        self.mono = mono
        self.duration = duration
    

    def forward(self,audio):
        audio, sr = librosa.load(audio,sr=self.sr,mono=self.mono,duration=self.duration)
        if len(audio) < sr:
            sil = np.zeros(sr-len(audio), dtype=float)
            audio = np.concatenate((audio,sil),axis=0)
        audio = audio[: 60 * sr]
        audio = torch.from_numpy(audio).to(torch.float32) # L,
        return audio


def preprocess(
    source: torch.Tensor,
    fbank_mean: float = 15.41663, # 预计算的Fbank均值
    fbank_std: float = 6.55582, # 预计算的Fbank标准差
) -> torch.Tensor: # 将原始音频转换为标准化的Fbank特征，供模型使用。
    fbanks = []
    for waveform in source: #每个音频分别处理 [32000] → [1, 32000] → Fbank → [T, 128]
        # 确保 waveform 至少有 400 个采样点（最好稍微大一点，比如 16000）
        if waveform.shape[-1] < 16000:
            import torch.nn.functional as F
            # 向后填充 0 直到长度为 400
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[-1]), "constant", 0)
        waveform = waveform.unsqueeze(0) * 2 ** 15 # 添加通道维度 [1, L]，因为Kaldi Fbank期望2D输入 /////// * 2 ** 15: 将归一化的音频值（-1到1）缩放到16-bit整数范围（-32768到32767）
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10) #输出128维Mel频率倒谱系数 //// 每帧25毫秒 → 400个采样点 (25ms × 16kHz)  ///// 帧移10毫秒 → 160个采样点 (10ms × 16kHz)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std) # 使用预计算的全局统计量进行标准化
    return fbank
# '''
# 音频长度 = 2.0 秒
# 帧长 = 0.025 秒
# 帧移 = 0.010 秒

# 原始音频: [------------------------------------------------] (2秒, 32000点)
#           ↓ 分帧处理
# 帧1:      [==========] 0-25ms (400点)
# 帧2:            [==========] 10-35ms (与前帧重叠15ms)  
# 帧3:                  [==========] 20-45ms
# ...
# 帧198:                                      [==========] 1975-2000ms
# 关键点：相邻帧有15ms的重叠，这确保了频谱变化的平滑性！

# 时间帧数 T = (音频长度 - 帧长) / 帧移 + 1
#            = (2.0 - 0.025) / 0.010 + 1
#            = 1.975 / 0.010 + 1
#            = 197.5 + 1 ≈ 198 帧

# 所以输出形状: [batch_size, 198, 128]
# '''



