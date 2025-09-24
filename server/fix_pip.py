"""
라이브러리 충돌 오류 해결
"""
!pip uninstall -y opencv-python opencv-python-headless numpy
!pip install numpy==1.26.4
!pip install opencv-python-headless==4.9.0.80
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install fastapi uvicorn pyngrok python-multipart nest_asyncio
