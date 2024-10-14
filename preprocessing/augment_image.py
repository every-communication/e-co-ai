import os
import cv2
import numpy as np

def augment_image(image):
    # 이미지 회전
    angle = np.random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # 이미지 이동
    tx = np.random.randint(-20, 20)
    ty = np.random.randint(-20, 20)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (w, h))

    # 이미지 뒤집기
    flipped_image = cv2.flip(translated_image, 1)  # 1: 좌우 뒤집기

    return flipped_image

def augment_video(video_path, output_dir, vocab, file_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []  # 증강된 프레임 저장할 리스트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 데이터 증강
        augmented_frame = augment_image(frame)
        
        # 증강된 프레임을 리스트에 저장
        frames.append(augmented_frame)
        frame_count += 1

    cap.release()

    # 증강된 프레임으로 비디오 저장
    if frames:
        height, width, layers = frames[0].shape
        output_video_path = os.path.join(output_dir, vocab, f"{file_name}_augmented.mp4")
        if not os.path.exists(os.path.dirname(output_video_path)):
            os.makedirs(os.path.dirname(output_video_path))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 비디오 작성 객체

        for frame in frames:
            out.write(frame)  # 프레임을 비디오로 저장

        out.release()  # 비디오 파일 닫기

def process_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for vocab in os.listdir(input_dir):
        vocab_path = os.path.join(input_dir, vocab)
        if os.path.isdir(vocab_path):
            for file in os.listdir(vocab_path):
                if file.endswith(('.mp4', '.mov')):
                    video_path = os.path.join(vocab_path, file)
                    print(f"Processing {video_path}...")
                    augment_video(video_path, output_dir, vocab, os.path.splitext(file)[0])

# 경로 설정
input_video_dir = 'dataset/processed_video'  # 원본 비디오 폴더 경로
output_augmented_dir = 'dataset/augmented_video'  # 증강된 비디오 저장 폴더 경로

# 비디오 증강 실행
process_videos(input_video_dir, output_augmented_dir)

print("Data augmentation and video reconstruction complete!")
