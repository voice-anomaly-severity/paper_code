import torch
import torch.nn as nn
import torchaudio


class SimpleStudentNet(nn.Module):
    def __init__(self, out_dim, sample_rate=16000, n_mels=80, layer_size=64):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, layer_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(layer_size, layer_size, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(layer_size, out_dim)

    def forward(self, wavs):
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        mels = self.melspec(wavs)
        mels = torch.log(mels + 1e-6)
        x = self.encoder(mels)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class DeepStudentNet(nn.Module):
    def __init__(self, out_dim, sample_rate=16000, n_mels=80, layer_size=128, depth=4):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_mels, layer_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(layer_size, kernel_size=5, dilation=2**i) for i in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(layer_size, out_dim)

    def forward(self, wavs):
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        mels = self.melspec(wavs)
        mels = torch.log(mels + 1e-6)
        x = self.input_conv(mels)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x


class StatsStudentNet(nn.Module):
    def __init__(self, out_dim, sample_rate=16000, n_mels=80, layer_size=96, dropout=0.1):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, layer_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Conv1d(layer_size, layer_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(layer_size * 2, out_dim)

    def forward(self, wavs):
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        mels = self.melspec(wavs)
        mels = torch.log(mels + 1e-6)
        x = self.encoder(mels)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        stats = torch.cat([mean, std], dim=-1)
        stats = self.dropout(stats)
        return self.proj(stats)
