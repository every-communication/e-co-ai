import os
import cv2
import numpy as np

# 원본 비디오와 저장할 경로 설정
video_dir = 'dataset/raw_video'
output_dir = 'dataset/processed_video'
resize_dim = 1080  # 정사각형 크기 (예: 1080x1080)

# 출력 경로가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def normalize_frame(frame):
    return (frame * 255).astype(np.uint8)

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # 원본 비디오의 프레임 속도 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {video_path} with FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 영상 프레임을 지정한 해상도로 리사이즈 (정사각형)
        frame = cv2.resize(frame, (resize_dim, resize_dim), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)

    cap.release()

    # 비디오 쓰기 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (resize_dim, resize_dim))

    for frame in frames:
        normalized_frame = normalize_frame(frame / 255.0)  # 정규화 후 [0, 255]로 변환
        out.write(normalized_frame)

    out.release()
    print(f"Saved normalized video to {output_path}")

def convert_mov_to_mp4(input_file, output_file):
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

def process_all_videos(video_dir, output_dir):
    if not os.path.exists(video_dir):
        print(f"Error: The directory '{video_dir}' does not exist.")
        return

    for vocab in os.listdir(video_dir):
        vocab_path = os.path.join(video_dir, vocab)
        output_vocab_path = os.path.join(output_dir, vocab)

        if not os.path.exists(output_vocab_path):
            os.makedirs(output_vocab_path)

        for video_file in os.listdir(vocab_path):
            if video_file.endswith('.mov') or video_file.endswith('.mp4'):
                input_file = os.path.join(vocab_path, video_file)
                output_file = os.path.join(vocab_path, f"{os.path.splitext(video_file)[0]}.mp4")

                # MOV 파일을 MP4로 변환
                if video_file.endswith('.mov'):
                    convert_mov_to_mp4(input_file, output_file)

                # 변환한 MP4 파일을 정규화하고 리사이즈
                normalized_output_path = os.path.join(output_vocab_path, f"{os.path.splitext(video_file)[0]}.mp4")
                process_video(output_file, normalized_output_path)

if __name__ == '__main__':
    process_all_videos(video_dir, output_dir)
