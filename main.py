from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import os
from dotenv import load_dotenv

from model import voice_detector
from utils import decode_base64_audio, extract_audio_features

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key-here")

# Initialize FastAPI
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated vs Human voices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audioFormat: Literal["mp3"]
    audioBase64: str = Field(..., min_length=100)
    
    @validator('audioBase64')
    def validate_base64(cls, v):
        if len(v) < 100:
            raise ValueError("Audio Base64 string too short")
        return v

# Response Model
class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str

class ErrorResponse(BaseModel):
    status: str
    message: str

# API Key Validation
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

# Main Endpoint
@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if a voice sample is AI-generated or human
    
    - **language**: One of Tamil, English, Hindi, Malayalam, Telugu
    - **audioFormat**: mp3
    - **audioBase64**: Base64-encoded MP3 audio file
    """
    try:
        # Decode audio
        waveform, sample_rate = decode_base64_audio(request.audioBase64)
        
        # Extract features
        features, waveform, sr = extract_audio_features(waveform, sample_rate)
        
        # Run prediction
        classification, confidence, explanation = voice_detector.predict(
            waveform, sr
        )
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "AI Voice Detection API",
        "status": "active",
        "endpoints": {
            "detection": "/api/voice-detection",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
