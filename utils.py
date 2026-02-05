import base64
import io
import librosa
import soundfile as sf
import numpy as np

def decode_base64_audio(base64_string: str) -> tuple:
    """
    Decode Base64 string to audio waveform
    Returns: (waveform, sample_rate)
    """
    try:
        # Decode Base64
        audio_bytes = base64.b64decode(base64_string)
        
        # Load audio from bytes
        audio_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_io)
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        
        return waveform, sample_rate
        
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {str(e)}")

def extract_audio_features(waveform: np.ndarray, sr: int) -> tuple:
    """
    Extract acoustic features from audio
    Returns: (features_dict, waveform, sample_rate)
    """
    # Resample to 16kHz (standard for speech models)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    features = {}
    
    # Spectral features
    features['spectral_centroid'] = np.mean(
        librosa.feature.spectral_centroid(y=waveform, sr=sr)
    )
    features['spectral_rolloff'] = np.mean(
        librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    )
    features['zero_crossing_rate'] = np.mean(
        librosa.feature.zero_crossing_rate(waveform)
    )
    
    # MFCCs (mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    return features, waveform, sr
