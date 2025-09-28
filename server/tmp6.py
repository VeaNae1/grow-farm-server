"""
결과 조회 api추가 // DB 버전 (더미데이터로 테스트용)
"""
!pip install fastapi uvicorn pyngrok "opencv-python-headless<4.3" python-multipart torch torchvision torchaudio nest_asyncio sqlalchemy

import os
import torch
import cv2
import numpy as np
import nest_asyncio
import secrets
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from PIL import Image as PILImage
import uvicorn

# DB 관련 import
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 환경 변수 설정
os.environ["NGROK_AUTH_TOKEN"] = "2jGzT5KrMCTO5lJZq68sIhvwo2N_41kScxnYWbstYACqQqjHS"  # ngrok 토큰
os.environ["AI_URL"] = ""  # 앱에서 호출할 URL (ngrok 주소가 들어갈 예정)

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
AI_URL = os.environ.get("AI_URL")

# ngrok 인증
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# DB 설정 (SQLite 사용)
DATABASE_URL = "sqlite:///./results.db"  # SQLite 파일 생성 (Colab 환경에서 로컬에 저장됨)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})  
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB 테이블 정의
class DetectionResult(Base):
    __tablename__ = "detection_results"
    id = Column(Integer, primary_key=True, index=True)  # PK
    filename = Column(String, index=True)  # 저장된 이미지 파일명
    detections = Column(JSON)  # 탐지 결과 (JSON 형태)

# 테이블 생성
Base.metadata.create_all(bind=engine)

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI 앱
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# 4. 헬스 체크 API(서버 살아있는지)
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is alive"}

# detect API (더미데이터 + DB 저장)
@app.post("/detect")
async def detect(file: UploadFile = File(...), db: Session = Depends(get_db)):
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

    # DB에 저장
    db_item = DetectionResult(filename=filename, detections=detections)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    return JSONResponse(content={
        "status": "success",
        "message": "추론 성공",
        "instance_detected": len(detections),
        "instances": detections
    })

# 결과 조회 API (DB)
@app.get("/results")
async def get_results(db: Session = Depends(get_db)):
    """
    지금까지 업로드된 모든 탐지 결과를 DB에서 조회
    """
    results = db.query(DetectionResult).all()
    return JSONResponse(content={
        "status": "success",
        "results": [
            {"id": r.id, "filename": r.filename, "detections": r.detections}
            for r in results
        ]
    })

public_url = ngrok.connect(8000)
AI_URL = f"{public_url}/detect"
print("공용 URL:", AI_URL)

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)

"""
보안 비밀 설정(환경 변수 설정 부분 코드로 대체 가능)
-ngrok 토큰 값 붙여넣기
-AI_URL은 앱에서 post 요청 보낼 url
-DB는 SQLite 사용 (Colab에서는 파일로 저장됨)

- Colab 테스트용: ngrok + public_url 사용
- 실제 서버 배포: ngrok 제거, 서버 도메인/IP + 포트 -> AI_URL 사용
- DB 저장 (memory_storage 대신 DB에 저장)
- 모바일 앱에서 /detect POST 요청으로 이미지 전달 가능
"""
