# -*- coding: utf-8 -*-
from ultralytics import YOLO
import argparse
import os

def train_model(data_yaml_path, epochs, image_size, base_model):
    """
    YOLOv8 모델을 학습시키는 함수

    :param data_yaml_path: 데이터셋의 정보가 담긴 data.yaml 파일 경로
    :param epochs: 총 학습 반복 횟수
    :param image_size: 학습에 사용할 이미지 크기
    :param base_model: 학습을 시작할 기반 모델 (e.g., 'yolov8n.pt')
    """
    print("--- YOLOv8 모델 학습을 시작합니다. ---")
    
    # 1. 학습을 시작할 기반 모델을 불러옵니다.
    # 'yolov8n.pt'는 가장 작고 빠른 모델로, 커스텀 학습을 시작하기에 좋습니다.
    # 만약 이전에 학습하던 모델이 있다면, 해당 .pt 파일 경로를 지정하여 이어서 학습할 수도 있습니다.
    try:
        model = YOLO(base_model)
        print(f"'{base_model}' 모델을 기반으로 학습을 시작합니다.")
    except Exception as e:
        print(f"오류: 기반 모델 '{base_model}'을 불러오는 데 실패했습니다. 파일이 존재하는지 확인하세요. \n{e}")
        return

    # 2. 우리만의 데이터셋으로 모델을 학습시킵니다.
    # 이 과정은 GPU 사용 시 수 분에서 수 시간이 걸릴 수 있습니다.
    try:
        print(f"데이터셋 설정 파일: {data_yaml_path}")
        print(f"학습 횟수 (Epochs): {epochs}")
        print(f"이미지 크기 (Image Size): {image_size}")
        print("학습 중에는 터미널에 진행 상황이 표시됩니다...")

        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            patience=20,  # 20번의 epoch 동안 성능 향상이 없으면 조기 종료
            batch=16,     # 한 번에 처리할 이미지 수 (GPU 메모리에 따라 조절)
            project="runs/train", # 학습 결과가 저장될 기본 폴더
            name="shot_detection_exp" # 실험 이름 (runs/train/shot_detection_exp 폴더 생성)
        )

        print("\n--- 모델 학습이 성공적으로 완료되었습니다! ---")
        print("학습 결과는 'runs/train/shot_detection_exp/weights' 폴더에서 확인할 수 있습니다.")
        print("가장 성능이 좋은 모델은 'best.pt' 파일로 저장됩니다.")

    except Exception as e:
        print(f"\n오류: 모델 학습 중 문제가 발생했습니다. \n{e}")
        print("data.yaml 파일의 경로와 내용이 올바른지 다시 한번 확인해주세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 모델 학습 스크립트")
    
    parser.add_argument(
        "--data", 
        required=True, 
        help="학습 데이터셋의 'data.yaml' 파일 경로"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="총 학습 반복 횟수 (epochs)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="학습에 사용할 이미지 크기"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="학습을 시작할 기반 모델 파일 경로"
    )
    
    args = parser.parse_args()

    # data.yaml 파일이 실제로 존재하는지 확인
    if not os.path.exists(args.data):
        print(f"오류: 지정된 경로에서 data.yaml 파일을 찾을 수 없습니다: {args.data}")
    else:
        train_model(args.data, args.epochs, args.imgsz, args.model)
