# SW-MAIN/deeplearning/real_time/real_time_inference_improved_v5.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

from deeplearning.models.model import CNNAE_LSTM_Transformer

def init_person_detector():
    """
    MobileNet SSD를 사용하여 영상에서 사람을 검출하는 함수.
    """
    proto_path = os.path.join("deeplearning", "models", "dnn", "MobileNetSSD_deploy.prototxt")
    model_path = os.path.join("deeplearning", "models", "dnn", "MobileNetSSD_deploy.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net

def detect_person(net, frame, conf_threshold=0.5):
    """
    입력 프레임에서 MobileNet SSD를 이용하여 사람 객체(박스)를 검출 후 목록 반환.
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # 'person' 클래스
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))
    return boxes

def preprocess_frame(frame, target_size=(224, 224), transform_pipeline=None):
    """
    OpenCV의 BGR 프레임을 PIL 이미지로 변환 후, target_size로 리사이즈하고 transform 적용.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    pil_img = pil_img.resize(target_size)
    if transform_pipeline:
        return transform_pipeline(pil_img)
    else:
        return pil_img

def main():
    # 전처리 파이프라인: 학습 시 사용한 설정 그대로 적용
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 임계값 및 temporal smoothing 관련 변수
    threshold = 0.75              # Falling 판정을 위한 임계값
    buffer_window = 5             # 최근 5회 예측 평균 계산을 위한 버퍼
    prediction_buffer = []
    
    # 예측 결과 텍스트 표시 지속시간 (초)
    prediction_display_duration = 3.0  
    last_prediction_text = ""
    last_prediction_time = 0

    # ROI 유지 시간 (새 검출 없더라도 ROI 유지)
    detection_timeout = 3.0       
    last_detection_time = 0
    last_box = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 모델 생성 및 체크포인트 불러오기 (원래 이름으로 수정)
    model = CNNAE_LSTM_Transformer(
        ae_latent_dim=128,
        gru_hidden_dim=256,
        lstm_num_layers=2,  # 내부적으로 GRU 사용
        transformer_d_model=256,
        transformer_nhead=4,
        transformer_num_layers=1,
        num_classes=2,
        cnn_feature_dim=256
    ).to(device)
    
    # 원래 체크포인트 파일명 사용
    checkpoint_path = "cnn_ae_lstm_transformer_lightcnn_v5_seq30_epoch100.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    
    # MobileNet SSD를 이용한 사람 검출 초기화
    person_detector = init_person_detector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    seq_length = 30  # 시퀀스 길이 (30 프레임)
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        current_time = time.time()
        boxes = detect_person(person_detector, frame, conf_threshold=0.5)
        
        # 새로운 사람 검출이 있으면 ROI 업데이트
        if len(boxes) > 0:
            max_area = 0
            best_box = None
            for (startX, startY, endX, endY) in boxes:
                area = (endX - startX) * (endY - startY)
                if area > max_area:
                    max_area = area
                    best_box = (startX, startY, endX, endY)
            if best_box:
                last_box = best_box
                last_detection_time = current_time
        else:
            # 새 사람 검출이 없으면, 마지막 ROI 유지 시간 확인
            if current_time - last_detection_time > detection_timeout:
                last_box = None
        
        # ROI가 있다면 사용하여 프레임 전처리
        if last_box is not None:
            (startX, startY, endX, endY) = last_box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
            processed_frame = preprocess_frame(frame, target_size=(224,224), transform_pipeline=transform_pipeline)
            frame_buffer.append(processed_frame)
        else:
            cv2.putText(frame, "No Person", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            frame_buffer = []
            prediction_buffer = []
        
        if len(frame_buffer) == seq_length:
            seq_tensor = torch.stack(frame_buffer, dim=0).unsqueeze(0).to(device)
            start_t = time.time()
            with torch.no_grad():
                logits, _, _ = model(seq_tensor)
            infer_time = time.time() - start_t
            prob = torch.softmax(logits, dim=1)
            fall_prob = prob[0, 1].item()
            prediction_buffer.append(fall_prob)
            if len(prediction_buffer) > buffer_window:
                prediction_buffer.pop(0)
            avg_fall_prob = np.mean(prediction_buffer)
            final_pred = 1 if avg_fall_prob >= threshold else 0
            if final_pred == 1:
                new_text = f"Falling (avg={avg_fall_prob:.2f}) [Inf:{infer_time*1000:.2f}ms]"
            else:
                new_text = f"Not Falling (avg={avg_fall_prob:.2f}) [Inf:{infer_time*1000:.2f}ms]"
            last_prediction_text = new_text
            last_prediction_time = time.time()
            frame_buffer = []
        
        # 예측 결과 텍스트를 최근 prediction_display_duration 동안 유지
        if time.time() - last_prediction_time < prediction_display_duration:
            cv2.putText(frame, last_prediction_text, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow("Real-time Fall Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
