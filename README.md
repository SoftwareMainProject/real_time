# 참고사항 
1. cnn_ae_lstm_transformer_lightcnn_v5_seq30_epoch100.pth 
파일명 함부로 수정하면 오류떠서 파일명에 lstm으로 보여도 실제론 gru 사용한 것임
대용량 파일이라 Git LFS 사용함. 혹은 직접 다운받기
git lfs install && git lfs pull을 해야 실제 파라미터 파일을 받을 수 있음!!!!!!!

2. dataset_loader/falling_dataset.py
이건 실시간 테스트에서 전처리 확인용으로 사용
만약 실시간 코드에서 전처리 로직 등을 불러오는 경우, 데이터셋 로더 파일이 필요할 수 있음
실시간 추론 코드에서 직접 전처리를 정의한다면, 굳이 넣지 않아도 됨!

3. real_time_inference.py
실시간 추론 스크립트 
(웹캠 → MobileNetSSD → CNN+AE+GRU+Transformer 모델 → 결과 오버레이)

4. lightweight_cnn_v5.py, model.py 파일
일반적으로 실시간 추론 코드에서 모델 클래스를 직접 정의한게 아니라 분리되어 있으므로 포함시킴.

실시간 추론을 실행하는 방법: pip install -r requirements.txt
