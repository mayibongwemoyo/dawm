import torch
import torchaudio
import numpy as np

class AudioEffects1:
    @staticmethod
    def add_noise(audio, noise_std=0.1):
        """Add Gaussian noise to the audio."""
        noise = torch.randn_like(audio) * noise_std
        return audio + noise

    @staticmethod
    def mp3_compression(audio, bitrate=64):
        """Simulate MP3 compression by resampling and quantizing."""
        # Resample to simulate compression artifacts
        target_sr = 16000  # Resample to 16kHz
        resampled = torchaudio.functional.resample(audio, orig_freq=44100, new_freq=target_sr)
        # Quantize to simulate bitrate reduction
        quantized = torch.round(resampled * (2 ** (bitrate // 8)) / (2 ** (bitrate // 8))
        return quantized

    @staticmethod
    def resample(audio, target_sr=8000):
        """Resample audio to a lower sample rate."""
        return torchaudio.functional.resample(audio, orig_freq=44100, new_freq=target_sr)

    @staticmethod
    def lowpass_filter(audio, cutoff_freq=5000, sample_rate=16000):
        """Apply a lowpass filter to the audio."""
        # Design a simple lowpass filter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
        # Apply the filter
        filtered = scipy.signal.lfilter(b, a, audio.numpy())
        return torch.tensor(filtered, dtype=torch.float32)