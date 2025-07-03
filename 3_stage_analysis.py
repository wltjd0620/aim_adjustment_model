# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import sys
import os
from ultralytics import YOLO
from collections import Counter
from scipy.spatial.distance import cdist

def analyze_shot_group(shots, image_width, image_height):
    """
    탐지된 탄흔 그룹을 분석하여 패턴과 통계치를 반환합니다.
    (yolo-advanced-analysis-script.py의 함수를 기반으로 함)
    """
    if shots.shape[0] == 0:
        return None

    group_center = tuple(np.mean(shots, axis=0).astype(int))

    # 1발만 있을 경우 분산도 계산이 무의미하므로 '양호'로 간주
    if shots.shape[0] < 2:
        return {
            'group_center': group_center,
            'final_pattern_id': 0, # 'good_group'
            'shots_count': shots.shape[0]
        }
        
    std_dev_x = np.std(shots[:, 0]) / image_width
    std_dev_y = np.std(shots[:, 1]) / image_height
    
    scatter_threshold = 0.04 # 3발 기준이므로 임계값 소폭 하향 조정

    is_horizontally_scattered = std_dev_x > scatter_threshold
    is_vertically_scattered = std_dev_y > scatter_threshold

    final_pattern_id = 0 # 기본값: good_group
    if is_horizontally_scattered and is_vertically_scattered:
        final_pattern_id = 4 # combined_scatter
    elif is_vertically_scattered:
        final_pattern_id = 1 # vertical_scatter
    elif is_horizontally_scattered:
        final_pattern_id = 2 # horizontal_scatter
            
    return {
        'group_center': group_center,
        'final_pattern_id': final_pattern_id,
        'shots_count': shots.shape[0]
    }

def get_feedback_for_stage(analysis_result, model_names, target_center, pixels_to_cm, click_value):
    """
    각 단계의 분석 결과에 따라 사용자 피드백과 클리크 값을 생성합니다.
    """
    if not analysis_result:
        return "탄흔을 찾지 못했습니다.", None, None

    pattern_id = analysis_result['final_pattern_id']
    group_center = analysis_result['group_center']
    shots_count = analysis_result['shots_count']
    
    feedback_map = {
        0: ("양호한 탄착군", f"{shots_count}발의 탄착군이 잘 형성되었습니다. 영점 조절을 위해 클리크 값을 참고하세요."),
        1: ("호흡 불량 의심 (수직 분산)", f"{shots_count}발의 탄착군이 상하로 분산되었습니다. 격발 시 호흡 조절에 집중하세요."),
        2: ("조준선 불량 의심 (수평 분산)", f"{shots_count}발의 탄착군이 좌우로 분산되었습니다. 안정적인 조준을 연습하세요."),
        4: ("혼합 분산 의심", "호흡과 조준이 모두 불안정합니다. 기본 사격술 훈련이 필요합니다.")
    }
    
    title, advice = feedback_map.get(pattern_id, ("분석 불가", "패턴을 식별할 수 없습니다."))
    
    click_adjustment = None
    # 1발만 맞았거나, 너무 분산된 경우는 클리크 조절이 무의미
    if shots_count > 0 and pattern_id != 4: 
        group_center_cm_x = (group_center[0] - target_center[0]) * pixels_to_cm
        group_center_cm_y = -(group_center[1] - target_center[1]) * pixels_to_cm
        
        horizontal_clicks = -group_center_cm_x / click_value
        vertical_clicks = -group_center_cm_y / click_value
        h_direction = "우" if horizontal_clicks > 0 else "좌"
        v_direction = "상" if vertical_clicks > 0 else "하"
        
        click_adjustment = (
            f"▶ {h_direction}으로 {abs(horizontal_clicks):.1f} 클리크\n"
            f"▶ {v_direction}으로 {abs(vertical_clicks):.1f} 클리크"
        )
    
    full_feedback = f"진단: {advice}\n"
    if click_adjustment:
        full_feedback += f"\n[클리크 조절 값]\n{click_adjustment}"
    else:
        full_feedback += "\n(클리크 조절 값은 제공되지 않습니다.)"
        
    return title, full_feedback, group_center

def find_new_shots(all_current_shots, previous_shots_list):
    """
    이전 탄흔들을 제외한 새로운 탄흔들만 찾아내는 함수
    """
    if not previous_shots_list:
        return all_current_shots

    # 현재 탄흔과 이전 탄흔들 간의 거리 계산
    # all_current_shots: (N, 2), previous_shots_list: (M, 2) -> distance_matrix: (N, M)
    distance_matrix = cdist(all_current_shots, previous_shots_list)
    
    new_shots = []
    # 각 현재 탄흔에 대해, 가장 가까운 이전 탄흔과의 거리를 확인
    for i, shot in enumerate(all_current_shots):
        min_dist_to_previous = np.min(distance_matrix[i])
        
        # 거리가 특정 임계값(e.g., 15px)보다 크면 새로운 탄흔으로 간주
        if min_dist_to_previous > 15:
            new_shots.append(shot)
            
    return np.array(new_shots)

def main(args):
    """메인 실행 함수"""
    try:
        model = YOLO(args.model)
        model_names = model.names
        print(f"'{args.model}' 모델을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: YOLO 모델을 불러오는 데 실패했습니다. \n{e}")
        sys.exit(1)

    previous_shots = [] # 이전 차수의 모든 탄흔을 기억하는 리스트

    # 각 차수(stage)의 이미지를 순서대로 처리
    for stage_idx, image_path in enumerate(args.images):
        print("\n" + "="*50)
        print(f"■■■ {stage_idx + 1}차 사격 분석 시작 (이미지: {image_path}) ■■■")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"오류: '{image_path}' 이미지를 열 수 없습니다. 다음 차수로 넘어갑니다.")
            continue

        h, w, _ = image.shape
        clone = image.copy()

        # 1. 모델을 통해 현재 이미지의 모든 탄흔 탐지
        results = model(image, classes=[0,1,2,3], verbose=False) # good_group, v_scatter, h_scatter 클래스만 탐지
        all_current_shots = np.array([box.xywh[0][:2].cpu().numpy() for box in results[0].boxes]) # 중심좌표만 추출

        if all_current_shots.shape[0] == 0:
            print("결과: 이미지에서 탄흔을 찾지 못했습니다.")
            continue
        
        # 2. 이전 탄흔들을 제외한 '새로운 탄흔'만 식별
        new_shots = find_new_shots(all_current_shots, previous_shots)
        
        print(f"총 {all_current_shots.shape[0]}개 탄흔 탐지, 그 중 새로운 탄흔: {new_shots.shape[0]}개")
        
        if new_shots.shape[0] == 0:
            print("결과: 새로운 탄흔이 없습니다. 이전 사격과 동일한 사진으로 보입니다.")
            # 현재 탐지된 모든 탄흔을 다음 차수를 위해 기억
            previous_shots.extend(all_current_shots)
            continue
            
        # 3. 새로운 탄흔들에 대한 분석 수행
        # 표적지 중심은 첫 번째 이미지의 중앙으로 가정
        target_center = (w//2, h//2)
        analysis_result = analyze_shot_group(new_shots, w, h)
        
        # 픽셀-cm 비율 계산 (임시: 10점 원 기준)
        pixels_to_cm = args.radius / (w * 0.2) # 10점원 반지름이 이미지 너비의 20%라고 가정

        title, feedback, group_center = get_feedback_for_stage(analysis_result, model_names, target_center, pixels_to_cm, args.click)
        
        print(feedback)
        
        # 4. 시각화
        # 이전 탄흔들은 회색으로 표시
        for shot in previous_shots:
            cv2.circle(clone, tuple(shot.astype(int)), 10, (150, 150, 150), 2)
        # 새로운 탄흔들은 빨간색으로 표시
        for shot in new_shots:
            cv2.circle(clone, tuple(shot.astype(int)), 10, (0, 0, 255), 2)
        
        # 새로운 탄착군의 중심은 녹색으로 표시
        if group_center:
            cv2.drawMarker(clone, group_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        cv2.putText(clone, f"Stage {stage_idx+1}: {title}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 결과 이미지 저장
        output_filename = f"stage_{stage_idx+1}_result.jpg"
        cv2.imwrite(output_filename, clone)
        print(f"분석 결과가 '{output_filename}' 으로 저장되었습니다.")

        # 5. 다음 차수를 위해 현재 탐지된 모든 탄흔을 기억
        previous_shots.extend(all_current_shots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Multi-Stage Shooting Analysis")
    parser.add_argument("-i", "--images", required=True, nargs='+', help="분석할 1~3차 표적지 이미지 경로 (순서대로)")
    parser.add_argument("-m", "--model", required=True, help="학습된 YOLOv8 모델 파일(.pt) 경로")
    parser.add_argument("-c", "--click", type=float, default=0.7, help="1클리크 당 이동 거리(cm)")
    parser.add_argument("-r", "--radius", type=float, default=5.0, help="표적지 10점 원의 실제 반지름(cm)")
    
    args = parser.parse_args()
    main(args)
