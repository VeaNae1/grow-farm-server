"""
최종
"""

# ---------------------------
# 콘솔 인코딩 오류 방지 설정 (Windows)
# ---------------------------
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# 표준 라이브러리
import pathlib
import subprocess
import time
from datetime import datetime
from collections import Counter
from statistics import mean

# 써드파티
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON, LargeBinary, ForeignKey, DateTime, or_
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from urllib.parse import quote_plus
from pyngrok import ngrok, conf
import requests
import uvicorn

# ---------------------------
# ngrok 설정
# ---------------------------
NGROK_AUTH_TOKEN = "2jGzT5KrMCTO5lJZq68sIhvwo2N_41kScxnYWbstYACqQqjHS"
os.environ["NGROK_AUTH_TOKEN"] = NGROK_AUTH_TOKEN
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
conf.get_default().encoding = "utf-8"  # ngrok 로그 인코딩 강제 설정

# ---------------------------
# MySQL 설정 (로컬)
# ---------------------------
db_user = "fastapi"
db_password = "Fastapi123@@"
db_host = "127.0.0.1"
db_port = 3306
db_name = "fastapi_test"

encoded_password = quote_plus(db_password)
DATABASE_URL = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"

# ---------------------------
# DB 엔진 및 세션 생성
# ---------------------------
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------
# 테이블 정의
# ---------------------------
class DetectionResult(Base):
    __tablename__ = "detection_results"
    id = Column(Integer, primary_key=True, index=True)
    main_image = Column(LargeBinary)
    # content:
    # {
    #   "total_leaves": int,
    #   "deficiency_prob": float,
    #   "model_accuracy": float (optional),
    #   "leaf_results":[{"label":str,"conf":float,"bbox":[x1,y1,x2,y2]}, ...]
    # }
    content = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    crops = relationship("PostCrop", back_populates="post", cascade="all, delete-orphan")

class PostCrop(Base):
    __tablename__ = "post_crops"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("detection_results.id"))
    crop_image = Column(LargeBinary)
    order_index = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    post = relationship("DetectionResult", back_populates="crops")

class RecommendationInfo(Base):
    __tablename__ = "recommendation_info"
    id = Column(Integer, primary_key=True, index=True)
    nutrient = Column(String(50), nullable=False)       # "N","P","K","healthy"
    fertilizer = Column(JSON)                           # ["...", ...]
    prevention = Column(JSON)                           # ["...", ...]
    symptoms = Column(JSON)                             # ["...", ...]
    type_code = Column(String(16), unique=True, index=True)  # N/P/K/healthy (있으면 사용)

# 테이블 없으면 생성
Base.metadata.create_all(bind=engine)

# ---------------------------
# DB 세션 의존성
# ---------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# 모델 로드
# ---------------------------
DETECT_WEIGHTS = "./leaf_detect.pt"
CLS_WEIGHTS    = "./lack_classify.pt"

print("📦 모델 로드 중...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_yolov5_model(weights_path: str):
    """Windows 환경 torch.hub + yolov5 로딩 이슈 우회"""
    try:
        return torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    except Exception as e1:
        try:
            _orig = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            m = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
            pathlib.PosixPath = _orig
            return m
        except Exception as e2:
            raise RuntimeError(f"YOLOv5 모델 로드 실패: {e1} / {e2}")

yolo_det = load_yolov5_model(DETECT_WEIGHTS)
cls_model = load_yolov5_model(CLS_WEIGHTS)

# 탐지 하이퍼파라미터(가능하면 적용)
try:
    yolo_det.conf = 0.6
    yolo_det.iou  = 0.45
    yolo_det.max_det = 50
except Exception:
    pass

yolo_det.to(device).eval()
cls_model.to(device).eval()
cls_names = getattr(cls_model, 'names', None) or ['healthy','n','p','k']
print("✅ 모델 로드 완료")

# ---------------------------
# 분류 전처리/추론 유틸
# ---------------------------
def preprocess_for_cls(crop_bgr, size=224):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    t = torch.from_numpy(img).unsqueeze(0).contiguous().to(device=device, dtype=torch.float32)
    return t

@torch.no_grad()
def classify_crop(crop_bgr):
    x = preprocess_for_cls(crop_bgr, size=224)
    out = cls_model(x)
    probs = None
    if hasattr(out, 'probs'):
        p = out.probs[0] if isinstance(out.probs, list) else out.probs
        probs = p if torch.is_tensor(p) else torch.tensor(p)
    if probs is None:
        logits = out if torch.is_tensor(out) else getattr(out, 'logits', torch.as_tensor(out))
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = F.softmax(logits[0], dim=-1)
    probs_np = probs.float().detach().cpu().numpy()
    idx = int(np.argmax(probs_np))
    if isinstance(cls_names, dict):
        label = cls_names.get(idx, f"class_{idx}")
    else:
        label = cls_names[idx] if idx < len(cls_names) else f"class_{idx}"
    return label, float(probs_np[idx])

# ---------------------------
# FastAPI 앱 설정
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------
# 서버 상태 확인
# ---------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is alive"}

# ---------------------------
# 감지 및 DB 저장
# ---------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="이미지 파일만 업로드 가능합니다.")

    image_bytes = await file.read()
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="이미지 디코딩 실패")

    det = yolo_det(img, size=416)
    df = det.pandas().xyxy[0]

    crops = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, conf = classify_crop(crop)
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue
        crops.append({
            "order_index": len(crops),
            "crop_blob": buf.tobytes(),
            "label": label,
            "conf": conf,
            "bbox": [x1, y1, x2, y2]
        })

    total = len(crops)
    lack_cnt = sum(1 for c in crops if str(c["label"]).lower() != "healthy")
    deficiency_prob = round((lack_cnt / total) * 100, 2) if total > 0 else 0.0

    # content에 모델 정확도 필드 포함(앱 표시용)
    model_accuracy = 93.0  # 필요 시 실제 값으로 교체
    db_main = DetectionResult(
        main_image=image_bytes,
        content={
            "total_leaves": total,
            "deficiency_prob": deficiency_prob,
            "model_accuracy": model_accuracy,
            "leaf_results": [
                {"label": c["label"], "conf": c["conf"], "bbox": c["bbox"]}
                for c in crops
            ]
        }
    )
    db.add(db_main)
    db.commit()
    db.refresh(db_main)

    crop_records = [
        PostCrop(post_id=db_main.id, crop_image=c["crop_blob"], order_index=c["order_index"])
        for c in crops
    ]
    if crop_records:
        db.add_all(crop_records)
        db.commit()

    return JSONResponse(content={
        "status": "success",
        "message": "탐지/분류 완료 및 DB 저장",
        "data": {
            "id": db_main.id,
            "total_leaves": total,
            "deficiency_prob": deficiency_prob,
            "detections": [
                {"label": c["label"], "conf": round(c["conf"], 4), "bbox": c["bbox"]}
                for c in crops
            ]
        }
    })

# ---------------------------
# 앱에서 호출할 업로드 엔드포인트
# ---------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await detect(file=file, db=db)

# ---------------------------
# 유틸: 이미지 MIME 추정, 절대 URL 구성
# ---------------------------
def _guess_media_type(img_bytes: bytes) -> str:
    if not img_bytes or len(img_bytes) < 4:
        return "image/png"
    if img_bytes[0] == 0xFF and img_bytes[1] == 0xD8:
        return "image/jpeg"
    if img_bytes[0] == 0x89 and img_bytes[1] == 0x50 and img_bytes[2] == 0x4E and img_bytes[3] == 0x47:
        return "image/png"
    return "image/png"

def _weekday_kr(dt: datetime) -> str:
    return ["월","화","수","목","금","토","일"][dt.weekday()]

def _major_deficiency(leaf_results):
    labels = [str(lr.get("label", "")).lower() for lr in leaf_results]
    lack_only = [l for l in labels if l and l != "healthy"]
    if not lack_only:
        return "HEALTHY"
    return Counter(lack_only).most_common(1)[0][0].upper()

def _abs_url(path: str, public_url: str | None, request: Request) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if public_url:
        return f"{public_url.rstrip('/')}{path}"
    base = str(request.base_url).rstrip("/")
    return f"{base}{path}"

# ---------------------------
# DB 결과 조회 (리스트)
# ---------------------------
@app.get("/results")
async def get_results(db: Session = Depends(get_db), request: Request = None):
    rows = db.query(DetectionResult).order_by(DetectionResult.id.desc()).all()
    return JSONResponse(content={
        "status": "success",
        "results": [
            {
                "id": r.id,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": r.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                "content": r.content,
                "crop_count": len(r.crops),
                "main_image_url": _abs_url(f"/image/{r.id}", public_url, request) if request else f"/image/{r.id}"
            } for r in rows
        ]
    })

# ---------------------------
# 이미지 바이너리 응답
# ---------------------------
@app.get("/image/{post_id}")
async def get_main_image(post_id: int, db: Session = Depends(get_db)):
    r = db.get(DetectionResult, post_id)
    if not r or not r.main_image:
        raise HTTPException(404, "이미지 없음")
    return Response(content=r.main_image, media_type=_guess_media_type(r.main_image))

@app.get("/crop/{post_id}")
async def get_crop_image(post_id: int, index: int = Query(0, ge=0), db: Session = Depends(get_db)):
    r = db.get(DetectionResult, post_id)
    if not r or not r.crops or index >= len(r.crops):
        raise HTTPException(404, "크롭 이미지 없음")
    blob = sorted(r.crops, key=lambda c: c.order_index)[index].crop_image
    return Response(content=blob, media_type=_guess_media_type(blob))

# ---------------------------
# 보고서(화면용) JSON
# ---------------------------
@app.get("/report/{post_id}")
async def get_report(post_id: int, request: Request, db: Session = Depends(get_db)):
    r = db.get(DetectionResult, post_id)
    if not r:
        raise HTTPException(404, "결과 없음")

    content = r.content or {}
    leaf_results = content.get("leaf_results", [])
    total_objects = int(content.get("total_leaves", len(leaf_results)))
    total_detected = sum(1 for lr in leaf_results if str(lr.get("label","")).lower() != "healthy")
    deficiency_prob = float(content.get("deficiency_prob", round(100 * total_detected / max(total_objects,1), 2)))

    # accuracy: content에 있으면 우선, 없으면 평균 confidence(%)로 계산
    accuracy = float(content.get("model_accuracy", 0.0))
    if not accuracy:
        confs = [float(lr.get("conf", 0.0)) for lr in leaf_results if isinstance(lr, dict)]
        accuracy = round(mean(confs) * 100, 1) if confs else 0.0

    # 이번 결과의 주요 결핍 원소
    def_type = _major_deficiency(leaf_results)  # N/P/K/HEALTHY

    # recommendation_info: type_code 또는 nutrient로 조회
    rec = None
    if def_type in ("N","P","K","HEALTHY"):
        rec = db.query(RecommendationInfo).filter(
            or_(RecommendationInfo.type_code == def_type, RecommendationInfo.nutrient == def_type)
        ).first()

    # 날짜(생성일)
    now = r.created_at or datetime.utcnow()
    date_str = f"{now.year}년 {now.month}월 {now.day}일({_weekday_kr(now)})"

    # 크롭 URL(최대 5개)
    crop_urls = [_abs_url(f"/crop/{r.id}?index={i}", public_url, request) for i in range(min(len(r.crops), 5))]

    payload = {
        "date": date_str,
        "deficiency_type": def_type,                       # ← 클라가 그대로 사용
        "deficiency_prob": deficiency_prob,
        "total_detected": total_detected,
        "total_objects": total_objects,
        "main_image_url": _abs_url(f"/image/{r.id}", public_url, request),
        "crop_image_urls": crop_urls,

        "accuracy": accuracy,                              # ← 탑레벨 accuracy
        "metrics": { "map_05": None, "precision": None, "recall": None, "f1": None },

        # 이번 결과의 타입(def_type)에 맞춘 추천/예방/증상
        "fertilizer_recommend": (rec.fertilizer if rec and rec.fertilizer else []),
        "prevention": (rec.prevention if rec and rec.prevention else []),
        "symptoms": (rec.symptoms if rec and rec.symptoms else []),

        "detections": leaf_results
    }
    return JSONResponse(content={"status": "success", "report": payload})

@app.get("/report/latest")
async def get_latest_report(request: Request, db: Session = Depends(get_db)):
    latest = db.query(DetectionResult).order_by(DetectionResult.id.desc()).first()
    if not latest:
        raise HTTPException(404, "최근 결과 없음")
    return await get_report(latest.id, request, db)

# ---------------------------
# (선택) 최신 1건 간단 요약
# ---------------------------
@app.get("/posts/latest")
def get_latest_post(db: Session = Depends(get_db), request: Request = None):
    r = db.query(DetectionResult).order_by(DetectionResult.created_at.desc()).first()
    if not r:
        return {"status": "empty", "result": None}

    leaf_results = (r.content or {}).get("leaf_results", [])
    labels = [str(x.get("label","")).lower() for x in leaf_results if x.get("label")]
    lack_labels = [l for l in labels if l != "healthy"]
    top_lack = Counter(lack_labels).most_common(1)[0][0].upper() if lack_labels else "N/A"

    rec = None
    if top_lack not in ("N/A", "", None):
        rec = db.query(RecommendationInfo).filter(
            or_(RecommendationInfo.type_code == top_lack, RecommendationInfo.nutrient == top_lack)
        ).first()

    result = {
        "id": r.id,
        "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "deficiency_prob": (r.content or {}).get("deficiency_prob", 0),
        "deficiency_type": top_lack,
        "total_detected": len(lack_labels),
        "total_objects": (r.content or {}).get("total_leaves", 0),
        "main_image_url": _abs_url(f"/image/{r.id}", public_url, request) if request else f"/image/{r.id}",
        "crops": [
            {
                "id": c.id,
                "order_index": c.order_index,
                "url": _abs_url(f"/crop/{r.id}?index={c.order_index}", public_url, request) if request else f"/crop/{r.id}?index={c.order_index}"
            }
            for c in sorted(r.crops, key=lambda x: x.order_index or 0)
        ],
        "recommend": {
            "fertilizers": (rec.fertilizer if rec and rec.fertilizer else []),
            "preventions": (rec.prevention if rec and rec.prevention else []),
        }
    }
    return {"status": "success", "result": result}

# ---------------------------
# ngrok 실행
# ---------------------------
public_url = None
try:
    subprocess.Popen(["ngrok", "http", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    tunnel_info = requests.get("http://127.0.0.1:4040/api/tunnels").json()
    public_url = tunnel_info["tunnels"][0]["public_url"]
    print("공용 URL:", public_url)
except Exception as e:
    print("ngrok URL을 가져오지 못했습니다:", e)
    public_url = None

# ---------------------------
# 서버 실행
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
