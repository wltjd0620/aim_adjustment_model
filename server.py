# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import json
import os
from datetime import datetime

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 데이터베이스 파일 경로 설정
DB_FILE = 'shot_records.json'

def load_database():
    """
    JSON 파일에서 데이터베이스를 불러옵니다. 파일이 없으면 빈 리스트를 반환합니다.
    """
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            # 파일이 비어있거나 형식이 잘못된 경우
            return []

def save_database(data):
    """
    데이터베이스를 JSON 파일에 저장합니다.
    """
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        # ensure_ascii=False로 한글이 깨지지 않게 하고, indent=4로 보기 좋게 저장
        json.dump(data, f, ensure_ascii=False, indent=4)


@app.route('/')
def index():
    """
    서버의 루트 URL로 접속했을 때, 저장된 모든 사격 기록을 보여줍니다.
    """
    records = load_database()
    # 최신 기록이 위로 오도록 역순으로 정렬
    return jsonify(sorted(records, key=lambda x: x['timestamp'], reverse=True))


@app.route('/log', methods=['POST'])
def log_shot_data():
    """
    '/log' 경로로 POST 요청을 받았을 때, 분석 결과를 받아서 저장합니다.
    """
    # 1. 클라이언트(분석 스크립트)가 보낸 JSON 데이터 추출
    new_record = request.json
    if not new_record:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    # 2. 현재 데이터베이스 불러오기
    db = load_database()

    # 3. 새로운 기록에 서버 시간 기준으로 타임스탬프 추가
    new_record['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 4. 데이터베이스에 새로운 기록 추가
    db.append(new_record)
    
    # 5. 변경된 데이터베이스 저장
    save_database(db)

    print(f"새로운 사격 기록이 추가되었습니다: {new_record['shooter_id']} at {new_record['timestamp']}")
    
    return jsonify({"status": "success", "message": "Record added."}), 201


if __name__ == '__main__':
    print("--- 데이터베이스 서버를 시작합니다. ---")
    print("서버 주소: http://127.0.0.1:5001")
    print("저장된 기록 확인: http://127.0.0.1:5001 접속")
    print("데이터 수신 경로: /log (POST 요청)")
    # host='0.0.0.0'으로 설정하면 같은 네트워크의 다른 기기에서도 접속 가능
    app.run(host='0.0.0.0', port=5001, debug=True)
