# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import sys
import os
import requests # 서버 통신을 위한 라이브러리
from ultralytics import YOLO
from datetime import datetime
from scipy.spatial.distance import cdist

# --- 서버 전송 함수 ---
def send_to_server(payload, server_url="http://127.0.0.1:5001/log"):
    """
    분석 결과를 JSON 형태로 서버에 전송합니다.
    """
    try:
        # 서버의 /log 엔드포인트에 POST 요청을 보냄
        response = requests.post(server_url, json=payload, timeout=5)
        
        # HTTP 상태 코드로 성공 여부 확인
        if response.status_code == 201:
            print(">> 분석 결과를 서버에 성공적으로 전송했습니다.")
        else:
            print(f">> 서버 전송 실패: 상태 코드 {response.status_code}, 메시지: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f">> 서버 연결 오류: 서버가 실행 중인지 확인하세요. ({e})")


# --- 기존 분석 함수들 (변경 없음) ---
def analyze_shot_group(shots, image_width, image_height):
    if shots.shape[0] == 0: return None
    group_center = tuple(np.mean(shots, axis=0).astype(int))
    if shots.shape[0] < 2: return {'group_center': group_center, 'final_pattern_id': 0, 'shots_count': shots.shape[0]}
    std_dev_x = np.std(shots[:, 0]) / image_width
    std_dev_y = np.std(shots[:, 1]) / image_height
    scatter_threshold = 0.04
    is_horizontally_scattered = std_dev_x > scatter_threshold
    is_vertically_scattered = std_dev_y > scatter_threshold
    final_pattern_id = 0
    if is_horizontally_scattered and is_vertically_scattered: final_pattern_id = 4
    elif is_vertically_scattered: final_pattern_id = 1
    elif is_horizontally_scattered: final_pattern_id = 2
    return {'group_center': group_center, 'final_pattern_id': final_pattern_id, 'shots_count': shots.shape[0]}

def get_feedback_for_stage(analysis_result):
    if not analysis_result: return "탄흔 없음", "분석 불가", None
    pattern_id = analysis_result['final_pattern_id']
    shots_count = analysis_result['shots_count']
    feedback_map = {
        0: ("양호한 탄착군", f"{shots_count}발의 탄착군이 잘 형성되었습니다."),
        1: ("호흡 불량 의심", f"{shots_count}발의 탄착군이 상하로 분산되었습니다."),
        2: ("조준선 불량 의심", f"{shots_count}발의 탄착군이 좌우로 분산되었습니다."),
        4: ("혼합 분산 의심", "호흡과 조준이 모두 불안정합니다.")
    }
    title, advice = feedback_map.get(pattern_id, ("분석 불가", "패턴 식별 불가"))
    return title, advice

def find_new_shots(all_current_shots, previous_shots_list):
    if not previous_shots_list: return all_current_shots
    distance_matrix = cdist(all_current_shots, previous_shots_list)
    new_shots = []
    for i, shot in enumerate(all_current_shots):
        if np.min(distance_matrix[i]) > 15:
            new_shots.append(shot)
    return np.array(new_shots)


# --- 메인 실행 함수 (서버 전송 로직 추가) ---
def main(args):
    try:
        model = YOLO(args.model)
        print(f"'{args.model}' 모델을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: YOLO 모델을 불러오는 데 실패했습니다. \n{e}")
        sys.exit(1)

    # ★★★ 이번 사격 전체를 묶어줄 고유한 세션 ID 생성 ★★★
    session_id = f"{args.user}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"사격 세션이 시작되었습니다. (ID: {session_id})")
    
    previous_shots = []

    for stage_idx, image_path in enumerate(args.images):
        stage_num = stage_idx + 1
        print("\n" + "="*50)
        print(f"■■■ {stage_num}차 사격 분석 시작 (이미지: {image_path}) ■■■")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"오류: '{image_path}' 이미지를 열 수 없습니다.")
            continue

        h, w, _ = image.shape
        results = model(image, classes=[0,1,2,3], verbose=False)
        all_current_shots = np.array([box.xywh[0][:2].cpu().numpy() for box in results[0].boxes])
        
        new_shots = find_new_shots(all_current_shots, previous_shots)
        
        print(f"총 {all_current_shots.shape[0]}개 탄흔 탐지, 새로운 탄흔: {new_shots.shape[0]}개")
        
        if new_shots.shape[0] == 0:
            print("결과: 새로운 탄흔이 없습니다.")
            previous_shots = list(all_current_shots) if len(all_current_shots) > 0 else []
            continue

        analysis_result = analyze_shot_group(new_shots, w, h)
        target_center = (w//2, h//2)
        pixels_to_cm = args.radius / (w * 0.2)
        title, advice = get_feedback_for_stage(analysis_result)
        
        click_adjustment_text = "N/A"
        if analysis_result and analysis_result['final_pattern_id'] != 4:
            group_center = analysis_result['group_center']
            group_center_cm_x = (group_center[0] - target_center[0]) * pixels_to_cm
            group_center_cm_y = -(group_center[1] - target_center[1]) * pixels_to_cm
            h_clicks = -group_center_cm_x / args.click
            v_clicks = -group_center_cm_y / args.click
            h_dir = "우" if h_clicks > 0 else "좌"
            v_dir = "상" if v_clicks > 0 else "하"
            click_adjustment_text = f"{h_dir} {abs(h_clicks):.1f}, {v_dir} {abs(v_clicks):.1f} 클리크"

        print(f"진단: {advice}")
        print(f"클리크 조절: {click_adjustment_text}")

        # ★★★ 서버로 전송할 데이터 페이로드(payload) 구성 ★★★
        payload = {
            "shooter_id": args.user,
            "session_id": session_id,
            "stage": stage_num,
            "analysis_result": {
                "title": title,
                "advice": advice,
                "click_adjustment": click_adjustment_text
            },
            "raw_data": {
                "detected_new_shots": new_shots.shape[0],
                "total_shots_on_target": all_current_shots.shape[0]
            }
        }
        
        # ★★★ 구성된 데이터를 서버로 전송 ★★★
        send_to_server(payload)

        # (기존 시각화 및 저장 로직은 동일)
        # ...
        
        previous_shots.extend(all_current_shots)
    print("\n" + "="*50)
    print("모든 차수 분석이 완료되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Multi-Stage Analysis with Server Logging")
    parser.add_argument("-i", "--images", required=True, nargs='+', help="분석할 1~3차 표적지 이미지 경로 (순서대로)")
    parser.add_argument("-m", "--model", required=True, help="학습된 YOLOv8 모델 파일(.pt) 경로")
    parser.add_argument("-u", "--user", required=True, help="사격자 ID 또는 이름")
    parser.add_argument("-c", "--click", type=float, default=0.7, help="1클리크 당 이동 거리(cm)")
    parser.add_argument("-r", "--radius", type=float, default=5.0, help="표적지 10점 원의 실제 반지름(cm)")
    
    args = parser.parse_args()
    main(args)
