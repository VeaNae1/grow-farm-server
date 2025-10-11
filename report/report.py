from fpdf import FPDF
import requests

SERVER_URL = "https://f473e73222cc.ngrok-free.app/results" # 매번 변경
response = requests.get(SERVER_URL)
data = response.json()

pdf = FPDF()
pdf.add_page()

pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Detection Report", ln=True, align="C")

# 한글 폰트
pdf.add_font("NanumGothic", "", r"NanumGothic.ttf", uni=True)
pdf.set_font("NanumGothic", "", 14)

for r in data["results"]:
    pdf.cell(0, 10, f'ID: {r["id"]}, 파일명: {r["filename"]}', ln=True)
    for det in r["detections"]:
        pdf.cell(0, 10, f'  - {det["class"]}: {det["confidence"]:.2f} ({det["x_min"]},{det["y_min"]}~{det["x_max"]},{det["y_max"]})', ln=True)

pdf.output("detection_report.pdf")
print("PDF 보고서 저장 완료")
