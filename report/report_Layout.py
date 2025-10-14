from fpdf import FPDF
from datetime import datetime
import requests

# 더미 데이터
data = {
    "date": "20XX년 X월 X일(월)",
    "deficiency_prob": 66,
    "total_detected": 6,
    "total_objects": 9,
    "fertilizer_recommend": ["질소 비료", "인산 비료", "칼륨 비료"],
    "prevention": ["주기적 토양 점검", "잎 상태 관찰", "균형 잡힌 시비"],
    "symptoms_N": ["잎이 누렇게 변함", "성장이 느림", "줄기 약화"],
    "symptoms_P": ["잎이 검붉게 변함", "뿌리 성장 저하", "꽃 피는 시기 지연"]
}


# PDF 기본 설정
pdf = FPDF()
pdf.add_page()
pdf.add_font("NanumGothic", "", "NanumGothic.ttf")
pdf.add_font("NanumGothic", "B", "NanumGothicBold.ttf")
pdf.set_font("NanumGothic", "", 14)

# 제목
pdf.set_font("NanumGothic", "B", 22)
pdf.cell(0, 15, "<분석 결과>", align="C", ln=True)
pdf.ln(10)

# 날짜
pdf.set_font("NanumGothic", "", 13)
pdf.cell(0, 10, data["date"], align="R", ln=True)

# 메인 이미지 및 분석정보
# 이미지
main_x_img, main_y_img, main_w_img, main_h_img = 20, 40, 100, 100
pdf.rect(main_x_img, main_y_img, main_w_img, main_h_img)

# 분석 결과
pdf.set_xy(130, 45)
pdf.set_font("NanumGothic", "B", 14)
pdf.cell(0, 10, f"결핍확률 N {data['deficiency_prob']}%", ln=True)

pdf.set_x(130)
pdf.set_font("NanumGothic", "", 12)
pdf.cell(0, 8, f"결핍/총잎수 = {data['total_detected']}/{data['total_objects']}", ln=True)

pdf.ln(5)
pdf.set_x(130)
pdf.set_font("NanumGothic", "", 11)
pdf.multi_cell(70, 7, "[모델 성능 평가]\nPrecision —\nRecall —\nmAP@0.5 —\nF1 Score —")

pdf.ln(5)
pdf.set_x(130)
pdf.multi_cell(70, 6, "※ 결과가 50% 이하로 낮을 경우 재진단이 필요합니다.")

# 크롭 이미지
pdf.set_xy(25, 150)
crop_img = [" ", " ", " ", " ", " "]
for i, title in enumerate(crop_img):
    pdf.rect(20 + i*35, 150, 30, 30)

# 비료추천/예방법
pdf.set_xy(25, 190)
pdf.set_font("NanumGothic", "B", 14)
pdf.cell(0, 10, "비료추천.", ln=True)

pdf.set_font("NanumGothic", "", 12)
for item in data["fertilizer_recommend"]:
    pdf.cell(0, 8, f"• {item}", ln=True)

pdf.ln(5)
pdf.set_font("NanumGothic", "B", 14)
pdf.cell(0, 10, "예방법.", ln=True)

pdf.set_font("NanumGothic", "", 12)
for item in data["prevention"]:
    pdf.cell(0, 8, f"• {item}", ln=True)

# N/P 결핍 증상
pdf.add_page()
pdf.set_font("NanumGothic", "B", 16)
pdf.cell(0, 10, "N의 결핍 증상", ln=True)

pdf.set_font("NanumGothic", "", 12)
for s in data["symptoms_N"]:
    pdf.cell(0, 8, f"• {s}", ln=True)

start_x, start_y = 130, 30
for row in range(2):
    for col in range(2):
        pdf.rect(start_x + col*40, start_y + row*40, 30, 30)

pdf.ln(80)
pdf.set_font("NanumGothic", "B", 16)
pdf.cell(0, 10, "P의 결핍 증상", ln=True)

pdf.set_font("NanumGothic", "", 12)
for s in data["symptoms_P"]:
    pdf.cell(0, 8, f"• {s}", ln=True)

start_y = pdf.get_y() + 5
for row in range(2):
    for col in range(2):
        pdf.rect(130 + col*40, start_y + row*40, 30, 30)

pdf.output("report_Layout.pdf")
print("완료")
