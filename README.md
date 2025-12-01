1. 가상환경 생성

2. 패키지 설치
pip install -r requirements.txt

3. 실행 방법
    1) AI 서버 실행 (포트 5001)
    python ai_server.py
    2) 백엔드 서버 실행 (포트 5002)
    python main.py
    3) 프론트 실행 (포트 8000)
    python3 -m http.server 8000
    4) 브라우저에서 http://localhost:8000/cam.html 열기

4. 로그 확인
    ./watch_logs.sh  # 실시간 로그 모니터링
    또는
    tail -f main_server.log ai_server.log
