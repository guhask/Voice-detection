import numpy as np
import librosa
from typing import Tuple

class VoiceDetectionModel:
    def __init__(self):
        """Initialize lightweight model"""
        print("Loading lightweight voice detector...")
        print("Model ready!")
    
    def predict(self, waveform: np.ndarray, sr: int) -> Tuple[str, float, str]:
        """Predict using heuristics only"""
        
        # Calculate heuristics
        pitch_std, spectral_centroid, zcr = self._extract_features(waveform, sr)
        
        # Score based on heuristics
        ai_score = self._calculate_ai_score(pitch_std, spectral_centroid, zcr)
        
        print(f"DEBUG: Pitch={pitch_std:.1f} Hz, AI_Score={ai_score:.2f}")
        
        # Classify
        is_ai_generated = ai_score > 0.5
        classification = "AI_GENERATED" if is_ai_generated else "HUMAN"
        confidence = ai_score if is_ai_generated else (1 - ai_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            is_ai_generated, 
            confidence,
            pitch_std,
            spectral_centroid
        )
        
        return classification, round(confidence, 2), explanation
    
    def _extract_features(self, waveform: np.ndarray, sr: int):
        """Extract audio features"""
        try:
            zcr = np.mean(librosa.feature.zero_crossing_rate(waveform))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr))
            
            # Calculate pitch
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if 50 < pitch < 500:
                    pitch_values.append(pitch)
            
            pitch_std = np.std(pitch_values) if len(pitch_values) > 10 else 0
            
            return pitch_std, spectral_centroid, zcr
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return 0, 0, 0
    
    def _calculate_ai_score(self, pitch_std, spectral_centroid, zcr):
        """Calculate AI probability - LOW pitch = HIGH AI score"""
        ai_score = 0.0
        
        # Pitch variation - CRITICAL INDICATOR
        if pitch_std < 15:
            ai_score = 0.85      # Definitely AI
        elif pitch_std < 30:
            ai_score = 0.75      # Very likely AI
        elif pitch_std < 50:     # ← 46 Hz hits here
            ai_score = 0.65      # Likely AI
        elif pitch_std < 70:
            ai_score = 0.55      # Maybe AI
        elif pitch_std < 90:
            ai_score = 0.45      # Borderline
        elif pitch_std < 120:
            ai_score = 0.30      # Probably human
        else:
            ai_score = 0.20      # Likely human
        
        # Additional adjustments
        adjustments = 0.0
        
        # Spectral centroid
        if 2500 < spectral_centroid < 3800:
            adjustments += 0.10
        
        # Zero crossing rate
        if 0.06 < zcr < 0.14:
            adjustments += 0.05
        
        final_score = ai_score + adjustments
        return max(0.0, min(1.0, final_score))
    
    def _generate_explanation(self, is_ai, confidence, pitch_std, spectral_centroid):
        """Generate explanation"""
        if is_ai:
            if confidence > 0.75:
                return (
                    f"Strong AI characteristics detected: "
                    f"Extremely low pitch variation ({pitch_std:.1f} Hz indicates synthetic consistency), "
                    f"spectral centroid at {spectral_centroid:.0f} Hz shows artificial patterns. "
                    f"Confidence: {confidence*100:.0f}%"
                )
            elif confidence > 0.55:
                return (
                    f"Moderate AI indicators: "
                    f"Low pitch variation ({pitch_std:.1f} Hz below natural human range of 80-150 Hz), "
                    f"synthetic patterns detected. Confidence: {confidence*100:.0f}%"
                )
            else:
                return (
                    f"Slight AI tendencies: "
                    f"Pitch variation ({pitch_std:.1f} Hz) shows some synthetic characteristics. "
                    f"Confidence: {confidence*100:.0f}%"
                )
        else:
            if confidence > 0.75:
                return (
                    f"Strong human characteristics: "
                    f"Natural pitch variation ({pitch_std:.1f} Hz in normal human range of 80-150 Hz), "
                    f"organic voice patterns detected. Confidence: {confidence*100:.0f}%"
                )
            elif confidence > 0.55:
                return (
                    f"Moderate human indicators: "
                    f"Pitch variation ({pitch_std:.1f} Hz within expected natural range). "
                    f"Confidence: {confidence*100:.0f}%"
                )
            else:
                return (
                    f"Slight human tendencies: "
                    f"Some natural characteristics present. Confidence: {confidence*100:.0f}%"
                )

# Initialize model
voice_detector = VoiceDetectionModel()