import os
import pandas as pd
from tqdm import tqdm

# 디렉터리 경로 및 매핑 파일 저장 경로 설정
input_path = 'dataset/raw_video'
output_excel = 'preprocessing/directory_mapping.xlsx'

# 디렉터리 이름을 숫자로 매핑할 딕셔너리 및 매핑 결과 저장 리스트 초기화
word_mapping = {}
mapping_data = []

# 디렉터리 내의 각 폴더에 대해 고유한 숫자 매핑 생성 및 이름 변환
for idx, folder in enumerate(tqdm(os.listdir(input_path))):
    print(folder)
    # 매핑 딕셔너리에 저장 (기존 폴더 이름 : 숫자 매핑)
    word_mapping[folder] = idx
    
    # 기존 폴더 경로와 새 숫자 폴더 경로 설정
    old_path = os.path.join(input_path, folder)
    new_folder = f"{idx + 1:04d}"  # 숫자형 매핑을 문자열로 변환하여 사용
    new_path = os.path.join(input_path, new_folder)
    
    # 디렉터리 이름 변환
    os.rename(old_path, new_path)
    
    # 매핑 정보를 리스트에 저장
    mapping_data.append({'Original': folder, 'Mapped': new_folder})

# 매핑 정보를 DataFrame으로 변환 후 Excel 파일로 저장
df = pd.DataFrame(mapping_data)
df.to_excel(output_excel, index=False)

print(f"변환 완료 및 매핑 정보를 {output_excel}에 저장하였습니다.")
