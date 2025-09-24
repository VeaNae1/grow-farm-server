"""
결과 조회 api추가 // 더미데이터로 테스트용
"""
!pip install fastapi uvicorn pyngrok "opencv-python-headless<4.3" python-multipart torch torchvision torchaudio nest_asyncio

import os
import torch
import cv2
import numpy as np
import nest_asyncio
import secrets
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from PIL import Image as PILImage
import uvicorn

# 환경 변수
os.environ["NGROK_AUTH_TOKEN"] = "2jGzT5KrMCTO5lJZq68sIhvwo2N_41kScxnYWbstYACqQqjHS" # 토큰 복붙
os.environ["AI_URL"] = ""  # 앱에서 호출할 URL

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
AI_URL = os.environ.get("AI_URL")

# ngrok 인증
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# FastAPI 앱
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# YOLOv5 모델 로드
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt') # YOLOv5 모델 경로 path에

# 헬스 체크(서버 살아있는지)
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is alive"}

# 메모리 임시 저장용 리스트 (DB 없이 테스트용)
memory_storage = []

# detect API
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=404, detail="No file uploaded")
    if not file.content_type.startswith("image"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="이미지 파일만 업로드 가능합니다.")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="지원하지 않는 이미지 확장자입니다.")

    random_name = secrets.token_urlsafe(16)
    filename = f"/content/{random_name}.{ext}"
    image = PILImage.open(file.file)
    image.save(filename)

    # 더미데이터
    detections = [
        {"class": "person", "confidence": 0.98, "x_min": 50, "y_min": 100, "x_max": 200, "y_max": 400},
        {"class": "dog", "confidence": 0.87, "x_min": 250, "y_min": 120, "x_max": 400, "y_max": 380},
    ]

    # memory_storage에 탐지 결과 저장 (DB 없이 테스트용)
    memory_storage.append({
        "filename": filename,
        "detections": detections
    })

    return JSONResponse(content={"status": "success", "message": "추론 성공", "instance_detected": len(detections), "instances": detections})

# 결과 조회 API
@app.get("/results")
async def get_results():
    """
    지금까지 업로드된 모든 탐지 결과 반환
    """
    return JSONResponse(content={
        "status": "success",
        "results": memory_storage
    })

# Colab용 ngrok
public_url = ngrok.connect(8000)
AI_URL = f"{public_url}/detect"
print("공용 URL:", AI_URL)

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)

"""
보안 비밀 설정(환경 변수 설정 부분 코드로 대체 가능)
-ngrok토큰 값 붙여넣기
-AI_URL은 앱에서 post요청 보낼 url

- Colab 테스트용: ngrok + public_url 사용
- 실제 서버 배포: ngrok 제거, 서버 도메인/IP + 포트 -> AI_URL사용
- memory_storage에 탐지 결과 저장 (DB 없이 테스트용)
- 모바일 앱에서 /detect POST 요청으로 이미지 전달 가능
"""
