# SQLite 파일 확인 (코랩 확인용)
!ls -l ./results.db

"""
위 코드와 별개의 셀(코랩)
"""
# DB내용 직접 확인 (코랩 확인용)
import sqlite3
conn = sqlite3.connect('./results.db')
c = conn.cursor()
for row in c.execute('SELECT * FROM detection_results'):
    print(row)
conn.close()
