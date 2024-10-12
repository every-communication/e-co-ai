import os
import cv2

def crop_to_square(frame):
    height, width = frame.shape[:2]
    min_dim = min(height, width)
    top = (height - min_dim) // 2
    left = (width - min_dim) // 2
    return frame[top:top+min_dim, left:left+min_dim]

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get output size as 1080x1080
    out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1080))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to square and resize to 1080x1080
        cropped_frame = crop_to_square(frame)
        resized_frame = cv2.resize(cropped_frame, (1080, 1080))
        out.write(resized_frame)

    cap.release()
    out.release()

def process_all_videos(video_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for vocab in os.listdir(video_dir):
        vocab_path = os.path.join(video_dir, vocab)
        output_vocab_path = os.path.join(output_dir, vocab)
        
        if not os.path.exists(output_vocab_path):
            os.makedirs(output_vocab_path)
        
        for video_file in os.listdir(vocab_path):
            if video_file.endswith(".mp4"):
                input_path = os.path.join(vocab_path, video_file)
                output_path = os.path.join(output_vocab_path, video_file)
                
                print(f"Processing {video_file} ...")
                process_video(input_path, output_path)

if __name__ == "__main__":
    video_dir = "./videos"  # 원본 비디오가 있는 디렉토리
    output_dir = "./processed_video"  # 저장할 디렉토리
    process_all_videos(video_dir, output_dir)
