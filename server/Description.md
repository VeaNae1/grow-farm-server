# Flow Plot
[Mobile App]    
   │   
   │ POST /detect (Upload Image)   
   ▼   
[FastAPI Server]    
   │   
   │ YOLOv5 model inference   
   │ Save detection results to DB (using DB_URL)   
   ▼    
[Database]    
   │   
   │ Save / Lookup Results   
   ▼    
[FastAPI Server]    
   │   
   │ JSON Response Return   
   ▼   
[Mobile App]      
   │   
   │ Display detection results (bound box, object information, etc.)   
## Working on it
* Add a new file each time you modify it
* When completed, the rest except for the final file will be moved to the tmp folder separately
## Progress and Planning
* ~기본 프레임~
* ~테스트 프레임~
* ~배포의 프레임~
* ~더미 데이터로 서버 테스트~
* ~DB 테스트(코랩)~
* 외부 DB사용 구현
* ~앱과의 연결 프레임~
* 앱과의 연결 프레임 테스트
* 모델 적용 및 테스트 완료 시까지 진행
* 배포 프레임워크 참조, 결합
* 전체 코드 프로토타입 완성
* 테스트, 수정 및 최종 완료

## Error
* ~colab 라이브러리 충돌 문제로 테스트 불가능-> 해결 후 테스트 필요~
