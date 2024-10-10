import os
import zipfile
import shutil
from tqdm import tqdm

# 압축 파일 경로 및 재배치할 대상 폴더 경로 설정
zip_file = 'C:\\Users\\junb0\\Documents\\GitHub\\e-co-ai\\dataset\\total.zip'   # 압축 파일 경로
output_path = 'C:\\Users\\junb0\\Documents\\GitHub\\e-co-ai\\dataset\\raw_video'  # 파일을 재배치할 폴더 경로

print(f'rearranging videos from {zip_file}...')
zf = zipfile.ZipFile(zip_file)
videos = [f for f in zf.namelist() if f.endswith('.mp4') or f.endswith('.mov')]

video_vocabs = []
count = 0

for video in tqdm(videos):
    video_dir = video[:video.rfind('/')]
    video_name = video.split('/')[-1]

    video_vocab = video_name.split('_')[1]

    dst_path = f'{output_path}/{video_vocab}'

    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)

    zf.extract(video, dst_path)
    shutil.move(os.path.join(dst_path, video), os.path.join(dst_path, video_name))

    count += 1
    video_vocabs.append(video_vocab)
    
    print(f'{len(set(video_vocabs))} words found')
    print(f"done! total {count} videos filtered\n")
