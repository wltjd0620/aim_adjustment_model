# -*- coding: utf-8 -*-
import os
import random
from PIL import Image, ImageDraw
import numpy as np
import argparse
import sys
from math import floor

def create_target_image(width=640, height=640):
    """
    가상의 표적지 배경 이미지를 생성합니다.
    """
    img = Image.new('RGB', (width, height), color='#EAE5D9') # 사격장 표적지 느낌의 색
    draw = ImageDraw.Draw(img)
    center_x, center_y = width // 2, height // 2

    # 동심원 그리기
    for i in range(10, 0, -1):
        radius = i * 25
        color = 'black' if i < 5 else '#505050'
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=color,
            width=2
        )
    # 십자선
    draw.line((center_x, 0, center_x, height), fill='#A0A0A0', width=1)
    draw.line((0, center_y, width, center_y), fill='#A0A0A0', width=1)

    return img

def create_bullethole_image(size=20):
    """
    가상의 탄흔 이미지를 생성합니다. (투명 배경)
    """
    hole_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hole_img)
    
    # 불규칙한 모양을 만들기 위해 여러 개의 작은 원을 겹침
    for _ in range(15):
        offset_x = random.randint(-4, 4)
        offset_y = random.randint(-4, 4)
        r = random.randint(3, 6)
        draw.ellipse(
            (size//2 + offset_x - r, size//2 + offset_y - r, size//2 + offset_x + r, size//2 + offset_y + r),
            fill='black'
        )
    return hole_img

def generate_yolo_dataset(output_dir, num_images):
    """
    지정된 수량만큼 이미지와 라벨 파일을 생성하여 YOLO 데이터셋을 구축합니다.
    """
    print("--- YOLOv8 고급 데이터셋 생성을 시작합니다. ---")
    
    # 1. 폴더 구조 생성
    train_img_path = os.path.join(output_dir, 'train', 'images')
    train_lbl_path = os.path.join(output_dir, 'train', 'labels')
    valid_img_path = os.path.join(output_dir, 'valid', 'images')
    valid_lbl_path = os.path.join(output_dir, 'valid', 'labels')

    for path in [train_img_path, train_lbl_path, valid_img_path, valid_lbl_path]:
        os.makedirs(path, exist_ok=True)

    # 2. 기본 에셋 생성 및 패턴 정의
    target_bg = create_target_image()
    bullet_hole = create_bullethole_image()
    bg_w, bg_h = target_bg.size
    hole_w, hole_h = bullet_hole.size

    # ★★★ 패턴 정의: 각 패턴의 특성과 클래스 ID를 지정 ★★★
    PATTERNS = [
        {'id': 0, 'name': 'good_group', 'std_x_factor': 1.0, 'std_y_factor': 1.0},
        {'id': 1, 'name': 'vertical_scatter', 'std_x_factor': 0.7, 'std_y_factor': 4.0},
        {'id': 2, 'name': 'horizontal_scatter', 'std_x_factor': 4.0, 'std_y_factor': 0.7},
        {'id': 3, 'name': 'random_scatter', 'std_x_factor': 0, 'std_y_factor': 0} # 특별 케이스
    ]

    # 3. 데이터 생성 루프
    num_train = floor(num_images * 0.8) # 80%는 학습용
    
    for i in range(num_images):
        is_train = True if i < num_train else False
        img = target_bg.copy()
        labels = []
        
        # ★★★ 이미지마다 생성할 패턴을 무작위로 선택 ★★★
        current_pattern = random.choice(PATTERNS)
        pattern_id = current_pattern['id']
        
        num_holes = random.randint(7, 15)
        
        # 탄착군 중심을 랜덤하게 설정 (표적지 중앙 근처)
        group_center_x = int(np.random.normal(bg_w / 2, bg_w / 8))
        group_center_y = int(np.random.normal(bg_h / 2, bg_h / 8))

        for _ in range(num_holes):
            # 패턴 유형에 따라 탄흔 위치 생성 로직을 다르게 적용
            if current_pattern['name'] == 'random_scatter':
                # 완전 난사 패턴: 표적지 전체에 완전 무작위로 생성
                pos_x = random.randint(int(bg_w*0.1), int(bg_w*0.9) - hole_w)
                pos_y = random.randint(int(bg_h*0.1), int(bg_h*0.9) - hole_h)
            else:
                # 그 외 패턴들: 정규분포를 사용하되, 패턴에 따라 표준편차를 조절
                base_std_dev = 25
                std_x = base_std_dev * current_pattern['std_x_factor']
                std_y = base_std_dev * current_pattern['std_y_factor']
                
                pos_x = int(np.random.normal(group_center_x, std_x))
                pos_y = int(np.random.normal(group_center_y, std_y))

            # 이미지 경계를 벗어나지 않도록 좌표 조정
            pos_x = max(0, min(pos_x, bg_w - hole_w))
            pos_y = max(0, min(pos_y, bg_h - hole_h))
            
            # 표적지에 탄흔 이미지 합성
            img.paste(bullet_hole, (pos_x, pos_y), bullet_hole)
            
            # YOLO 라벨 포맷으로 변환. 클래스 ID는 선택된 패턴의 ID를 사용.
            x_center_norm = (pos_x + hole_w / 2) / bg_w
            y_center_norm = (pos_y + hole_h / 2) / bg_h
            width_norm = hole_w / bg_w
            height_norm = hole_h / bg_h
            
            labels.append(f"{pattern_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        # 4. 이미지 및 라벨 파일 저장
        filename = f"{current_pattern['name']}_{i:04d}" # 파일 이름에 패턴 종류 포함
        if is_train:
            img.save(os.path.join(train_img_path, f"{filename}.jpg"))
            with open(os.path.join(train_lbl_path, f"{filename}.txt"), "w") as f:
                f.write("\n".join(labels))
        else:
            img.save(os.path.join(valid_img_path, f"{filename}.jpg"))
            with open(os.path.join(valid_lbl_path, f"{filename}.txt"), "w") as f:
                f.write("\n".join(labels))

        sys.stdout.write(f"\r>> 생성 진행률: {i+1}/{num_images} (패턴: {current_pattern['name']})")
        sys.stdout.flush()

    # 5. ★★★ data.yaml 파일 생성 (클래스 정보 업데이트) ★★★
    class_names = [p['name'] for p in sorted(PATTERNS, key=lambda x: x['id'])]
    yaml_content = f"""
train: {os.path.abspath(train_img_path)}
val: {os.path.abspath(valid_img_path)}

# 클래스 개수
nc: {len(class_names)}

# 클래스 이름 (순서 중요)
names: {class_names}
"""
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"\n\n--- 데이터셋 생성 완료! ---")
    print(f"총 {num_images}개의 다양한 패턴 이미지와 라벨 파일이 생성되었습니다.")
    print(f"경로: {os.path.abspath(output_dir)}")
    print(f"\n이제 '{os.path.join(output_dir, 'data.yaml')}' 파일을 사용하여 모델 학습을 시작할 수 있습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8용 고급 가상 표적지 데이터셋 자동 생성기")
    parser.add_argument("-o", "--output", type=str, default="generated_dataset_advanced", help="생성된 데이터셋이 저장될 폴더 이름")
    parser.add_argument("-n", "--count", type=int, default=200, help="생성할 총 이미지의 개수 (학습/검증 포함)")
    args = parser.parse_args()
    
    generate_yolo_dataset(args.output, args.count)
