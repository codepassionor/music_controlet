import librosa
import numpy as np
from scipy.signal import butter, lfilter, savgol_filter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#torch.set_default_dtype(torch.float32)

def high_pass_filter(y, sr, cutoff=261.63, order=5):
    """高通滤波器，移除低频干扰。"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, y)

def process_audio(audio_path):
    """
    综合音频处理器，提取Chroma、Dynamics和PLP特征。
    参数:
        audio_path: 音频文件路径。
    返回:
        chroma: 原始Chroma矩阵
        chroma_one_hot: One-Hot编码的Chroma矩阵
        dynamics_db: 每帧动态特征（dB）
        plp: 节奏强度曲线
    """
    y, sr = librosa.load(audio_path)
    
    y_filtered = high_pass_filter(y, sr)
    chroma = librosa.feature.chroma_stft(y=y_filtered, sr=sr, n_chroma=12)
    chroma_one_hot = np.zeros_like(chroma)
    chroma_one_hot[np.argmax(chroma, axis=0), np.arange(chroma.shape[1])] = 1
    

    S = librosa.stft(y)
    frame_energy = np.sum(np.abs(S)**2, axis=0)
    dynamics_db = librosa.amplitude_to_db(frame_energy, ref=np.max)
    dynamics_db = savgol_filter(dynamics_db, window_length=11, polyorder=2)
    
    plp = librosa.beat.plp(y=y, sr=sr)
    
    return chroma, chroma_one_hot, dynamics_db, plp

class FeatureConverter(nn.Module):
    def __init__(self, input_length=7376, output_length=768, chroma_dim=12):
        """
        转换器类，包含三个独立的MLP模块，用于将输入特征映射到目标维度。
        
        参数：
        - input_length: 输入特征的时间维长度。
        - output_length: 输出特征的时间维长度。
        - chroma_dim: chroma_one_hot 的特征维长度。
        """
        super(FeatureConverter, self).__init__()
        self.chroma_mlp = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.ReLU()
        )
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.ReLU()
        )
        self.rhythm_mlp = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.ReLU()
        )
        self.chroma_dim = chroma_dim

    def forward(self, chroma_one_hot, dynamics_db, plp):
        """
        前向传播，处理输入特征并返回映射后的特征。
        
        参数：
        - chroma_one_hot: Chroma 特征，形状为 (chroma_dim, input_length)。
        - dynamics_db: Dynamics 特征，形状为 (input_length,)。
        - plp: Rhythm 特征，形状为 (input_length,)。
        
        返回：
        - processed_chroma: 映射后的 Chroma 特征，形状为 (chroma_dim, output_length)。
        - processed_dynamics: 映射后的 Dynamics 特征，形状为 (chroma_dim, output_length)。
        - processed_plp: 映射后的 Rhythm 特征，形状为 (chroma_dim, output_length)。
        """
        processed_chroma = torch.stack(
            [self.chroma_mlp(chroma_one_hot[i]) for i in range(self.chroma_dim)]
        )
        
        processed_dynamics = self.dynamics_mlp(dynamics_db).unsqueeze(0).expand(self.chroma_dim, -1)
        processed_plp = self.rhythm_mlp(plp).unsqueeze(0).expand(self.chroma_dim, -1)

        return processed_chroma, processed_dynamics, processed_plp

audio_path = 'data/train/A Classic Education - NightOwl.bass.mp3'
chroma, chroma_one_hot, dynamics_db, plp = process_audio(audio_path)

plt.figure(figsize=(16, 8))

plt.subplot(4, 1, 1)
plt.imshow(chroma, aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title("Chroma Features")

plt.subplot(4, 1, 2)
plt.imshow(chroma_one_hot, aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title("One Hot Chroma Features")

plt.subplot(4, 1, 3)
plt.plot(dynamics_db, label="Dynamics (dB)")
plt.legend()
plt.title("Dynamics Over Time")

plt.subplot(4, 1, 4)
plt.plot(plp, label="Rhythmic Intensity (PLP)")
plt.legend()
plt.title("Rhythmic Intensity")

plt.tight_layout()
plt.savefig('signal.png')

converter = FeatureConverter(input_length=plp.shape[0], output_length=768, chroma_dim=chroma_one_hot.shape[0])
melody = torch.from_numpy(chroma_one_hot).float()
dynamic = torch.from_numpy(dynamics_db).float()
rhythm = torch.from_numpy(plp).float()
processed_chroma, processed_dynamics, processed_plp = converter(melody, dynamic, rhythm)
