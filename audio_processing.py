import torch
import torchaudio
import torchaudio.transforms as T
import io

def preprocess_audio(audio_bytes, device):
    """
    Preprocess audio file for both LCNN and Spectrogram models
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_bytes)
    
    # Resample if needed (to 16kHz)
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Ensure minimum length (pad if necessary)
    min_length = 16000  # 1 second at 16kHz
    if waveform.size(1) < min_length:
        padding = torch.zeros(1, min_length - waveform.size(1))
        waveform = torch.cat((waveform, padding), dim=1)
    
    # Create spectrogram
    spectrogram_transform = T.Spectrogram(
        n_fft=512,
        win_length=512,
        hop_length=256,
        power=2.0
    )
    
    # Apply transformation
    spectrogram = spectrogram_transform(waveform)
    
    # Log spectrogram for better feature representation
    spectrogram = torch.log(spectrogram + 1e-9)
    
    # Move to device
    waveform = waveform.to(device)
    spectrogram = spectrogram.to(device)
    
    return waveform, spectrogram