import os
import zipfile
import shutil
from tqdm import tqdm

# 압축 파일 경로 및 재배치할 대상 폴더 경로 설정
zip_file = 'dataset/total.zip'   # 압축 파일 경로
output_path = 'dataset/raw_video'  # 파일을 재배치할 폴더 경로

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

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

    extracted_path = zf.extract(video, dst_path)
    shutil.move(extracted_path, os.path.join(dst_path, video_name))

    count += 1
    video_vocabs.append(video_vocab)
    
    # 추출된 경로에서 {사람이름} 및 {vocab} 디렉토리 삭제
    # 사람 이름 디렉토리 경로 생성
    person_name_dir = os.path.join(dst_path, video_name.split('_')[0].encode('utf-8').decode('utf-8'))  # {사람이름} 디렉토리

    # {사람이름} 디렉토리와 그 안의 내용을 삭제
    if os.path.exists(person_name_dir):
        shutil.rmtree(person_name_dir)
    
    print(f'{len(set(video_vocabs))} words found')
    print(f"done! total {count} videos filtered\n")
