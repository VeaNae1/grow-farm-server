"""
DB 연결
"""
!pip install fastapi uvicorn pyngrok "opencv-python-headless<4.3" python-multipart torch torchvision torchaudio sqlalchemy psycopg2-binary

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
from sqlalchemy import create_engine, text

# 환경 변수
os.environ["NGROK_AUTH_TOKEN"] = "ngrok_토큰" # 토큰 복붙
os.environ["AI_URL"] = ""  # 앱에서 호출할 URL
os.environ["DB_URL"] = "postgresql://user:pass@host:5432/dbname"
"""
postgresql	DB 종류 (PostgreSQL)
user 사용자 계정
pass	DB 비밀번호
host	DB 서버 주소 (localhost 또는 클라우드 호스트)
5432	DB 포트 (PostgreSQL 기본 5432)
dbname	접속할 DB 이름
으로 설정해 사용
"""

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
AI_URL = os.environ.get("AI_URL")
DB_URL = os.environ.get("DB_URL")

# ngrok 인증
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# DB 연결
engine = create_engine(DB_URL)

# FastAPI 앱
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt') # YOLOv5 모델 경로 path에

# 헬스 체크(서버 살아있는지)
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is alive"}

# detect API
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=404, detail="No file uploaded")
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미지 파일만 업로드 가능합니다.")
    
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="지원하지 않는 이미지 확장자입니다.")
    
    random_name = secrets.token_urlsafe(16)
    filename = f"/content/{random_name}.{ext}"
    image = PILImage.open(file.file)
    image.save(filename)
    
    img = cv2.imread(filename)
    results = model(img)
    detections = results.pandas().xyxy[0].to_dict(orient="records") # 각 객체 정보 반환
    
    # DB 저장
    with engine.connect() as conn:
        query = text("INSERT INTO detections (filename, result_json) VALUES (:f, :r)")
        conn.execute(query, {"f": filename, "r": str(detections)})
    
    return JSONResponse(content={"status": "success", "message": "추론 성공", "instance_detected": len(detections), "instances": detections})

# Colab용 ngrok
public_url = ngrok.connect(8000)
AI_URL = f"{public_url}/detect"
print("공용 URL:", AI_URL)

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000, block=False)

"""
보안 비밀 설정(환경 변수 설정 부분 코드로 대체 가능)
-ngrok토큰 값 붙여넣기
-AI_URL은 앱에서 post요청 보낼 url
-DB_URL은 서버의 탐지 결과, 로그, 사용자 데이터 저장용 url

- Colab 테스트용: ngrok + public_url 사용
- 실제 서버 배포: ngrok 제거, 서버 도메인/IP + 포트 -> AI_URL사용
- DB 저장 포함 (탐지 결과 저장 가능)
- 모바일 앱에서 /detect POST 요청으로 이미지 전달 가능
"""
