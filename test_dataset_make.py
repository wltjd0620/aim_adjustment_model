# -*- coding: utf-8 -*-
import os
import random
from PIL import Image, ImageDraw
import numpy as np
import argparse
import sys

def create_target_image(width=640, height=640):
    """
    가상의 표적지 배경 이미지를 생성합니다.
    """
    img = Image.new('RGB', (width, height), color='#EAE5D9')
    draw = ImageDraw.Draw(img)
    center_x, center_y = width // 2, height // 2

    for i in range(10, 0, -1):
        radius = i * 25
        color = 'black' if i < 5 else '#505050'
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=color,
            width=2
        )
    draw.line((center_x, 0, center_x, height), fill='#A0A0A0', width=1)
    draw.line((0, center_y, width, center_y), fill='#A0A0A0', width=1)
    return img

def create_bullethole_image(size=20):
    """
    가상의 탄흔 이미지를 생성합니다. (투명 배경)
    """
    hole_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hole_img)
    for _ in range(15):
        offset_x = random.randint(-4, 4)
        offset_y = random.randint(-4, 4)
        r = random.randint(3, 6)
        draw.ellipse(
            (size//2 + offset_x - r, size//2 + offset_y - r, size//2 + offset_x + r, size//2 + offset_y + r),
            fill='black'
        )
    return hole_img

def generate_test_stages(output_dir):
    """
    1, 2, 3차 사격 결과가 누적된 테스트 이미지를 생성합니다.
    """
    print(f"--- 다단계 테스트 이미지 생성을 시작합니다. ---")
    os.makedirs(output_dir, exist_ok=True)

    target_bg = create_target_image()
    bullet_hole = create_bullethole_image()
    bg_w, bg_h = target_bg.size
    hole_w, hole_h = bullet_hole.size

    # 클리크 조절을 시뮬레이션하기 위해, 탄착군 중심이 점점 표적지 중앙으로 이동
    stage_centers = [
        # Stage 1: 중앙에서 멀리 떨어진 곳
        (int(np.random.normal(bg_w/2, 80)), int(np.random.normal(bg_h/2, 80))),
        # Stage 2: 조금 더 가까워짐
        (int(np.random.normal(bg_w/2, 40)), int(np.random.normal(bg_h/2, 40))),
        # Stage 3: 중앙에 거의 근접
        (int(np.random.normal(bg_w/2, 20)), int(np.random.normal(bg_h/2, 20))),
    ]
    
    all_hole_positions = [] # 모든 차수의 탄흔 위치를 저장하는 리스트

    for i in range(3): # 3개 차수 생성
        stage_num = i + 1
        print(f"Stage {stage_num} 이미지 생성 중...")
        
        # 현재 차수의 탄착군 중심
        group_center_x, group_center_y = stage_centers[i]
        
        # 현재 차수에서 쏠 총알 수 (2 또는 3발)
        num_new_shots = random.choice([2, 3])
        
        for _ in range(num_new_shots):
            # 탄착군 중심 주변으로 탄흔 위치 결정 (분산도 25)
            pos_x = int(np.random.normal(group_center_x, 25))
            pos_y = int(np.random.normal(group_center_y, 25))
            
            # 이미지 경계를 벗어나지 않도록 좌표 조정
            pos_x = max(0, min(pos_x, bg_w - hole_w))
            pos_y = max(0, min(pos_y, bg_h - hole_h))
            
            all_hole_positions.append((pos_x, pos_y))

        # 현재 스테이지의 최종 이미지 생성
        stage_image = target_bg.copy()
        for pos in all_hole_positions:
            stage_image.paste(bullet_hole, pos, bullet_hole)
        
        # 이미지 파일 저장
        output_path = os.path.join(output_dir, f"stage_{stage_num}.jpg")
        stage_image.save(output_path)
        print(f"-> '{output_path}' 저장 완료. (총 {len(all_hole_positions)}발 누적)")
    
    print("\n--- 모든 테스트 이미지 생성 완료 ---")
    print(f"생성된 파일들은 '{os.path.abspath(output_dir)}' 폴더에서 확인할 수 있습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다단계 영점 사격 테스트 이미지 생성기")
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="test_images", 
        help="생성된 테스트 이미지가 저장될 폴더 이름"
    )
    args = parser.parse_args()
    generate_test_stages(args.output)
