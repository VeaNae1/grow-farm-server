"""
결과 조회 api추가 // DB 버전 (더미데이터로 테스트용, MySQL, VsCode에서 작동)
"""

# 콘솔 인코딩 오류 방지 설정 (Windows)
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import secrets
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from urllib.parse import quote_plus
from pyngrok import ngrok, conf
from PIL import Image as PILImage
import uvicorn
import subprocess
import requests
import time

NGROK_AUTH_TOKEN = "2jGzT5KrMCTO5lJZq68sIhvwo2N_41kScxnYWbstYACqQqjHS"
os.environ["NGROK_AUTH_TOKEN"] = NGROK_AUTH_TOKEN
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
conf.get_default().encoding = "utf-8"  # ngrok 로그 인코딩 강제 설정

# MySQL 설정 (로컬)
db_user = "fastapi"
db_password = "Fastapi123@@"
db_host = "127.0.0.1"
db_port = 3306
db_name = "fastapi_test"

encoded_password = quote_plus(db_password)
DATABASE_URL = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"

# DB 엔진 및 세션 생성
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DetectionResult(Base):
    __tablename__ = "detection_results"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    detections = Column(JSON)

# 테이블 없으면 생성
Base.metadata.create_all(bind=engine)

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI 앱 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "message": "Server is alive"}

@app.post("/detect")
async def detect(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """이미지 업로드 및 더미 감지 결과 반환"""
    if not file:
        raise HTTPException(status_code=404, detail="No file uploaded")
    if not file.content_type.startswith("image"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="이미지 파일만 업로드 가능합니다.")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="지원하지 않는 이미지 확장자입니다.")

    random_name = secrets.token_urlsafe(16)
    filename = f"./{random_name}.{ext}"
    image = PILImage.open(file.file)
    image.save(filename)

    # 더미 데이터
    detections = [
        {"class": "person", "confidence": 0.98, "x_min": 50, "y_min": 100, "x_max": 200, "y_max": 400},
        {"class": "dog", "confidence": 0.87, "x_min": 250, "y_min": 120, "x_max": 400, "y_max": 380},
    ]

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

@app.get("/results")
async def get_results(db: Session = Depends(get_db)):
    """DB에 저장된 감지 결과 전체 조회"""
    results = db.query(DetectionResult).all()
    return JSONResponse(content={
        "status": "success",
        "results": [
            {"id": r.id, "filename": r.filename, "detections": r.detections}
            for r in results
        ]
    })

# ngrok 실행
subprocess.Popen(["ngrok", "http", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ngrok 프로세스가 URL을 준비할 시간
time.sleep(3)

# ngrok이 실행 중인지 확인, public_url
try:
    tunnel_info = requests.get("http://127.0.0.1:4040/api/tunnels").json()
    public_url = tunnel_info["tunnels"][0]["public_url"]
    print("공용 URL:", public_url)
except Exception as e:
    print("ngrok URL을 가져오지 못했습니다:", e)
    public_url = None
    
# 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
