import os
import cv2

# 원본 비디오 경로 설정
input_dir = 'C:\\Users\\admin\\Documents\\GitHub\\e-co-ai\\dataset\\raw_video'

# 원본 디렉토리 내의 모든 MOV 파일 변환
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.mov'):
            input_file = os.path.join(root, file)
            output_file = os.path.join(root, f"{os.path.splitext(file)[0]}.mp4")

            # 비디오 캡처 및 저장
            cap = cv2.VideoCapture(input_file)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()
            print(f"Converted: {input_file} to {output_file}")

            # 원본 MOV 파일 삭제 (원하면 주석 해제)
            os.remove(input_file)
