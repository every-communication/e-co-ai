import numpy as np
import os
from torch.utils.data import Dataset
class SignLanguageDataset(Dataset):
    def __init__(self, keypoints_dir='C:\\Users\\admin\\Documents\\github\\e-co-ai\\dataset\\keypoints'):
        self.keypoints_dir = keypoints_dir
        self.data = self.load_data()
        self.label_to_index = self.create_label_to_index()  # 레이블을 인덱스로 매핑하는 사전 생성

    def load_data(self):
        data = []
        for vocab in os.listdir(self.keypoints_dir):
            vocab_path = os.path.join(self.keypoints_dir, vocab)
            for file in os.listdir(vocab_path):
                if file.endswith('.npy'):
                    keypoint_path = os.path.join(vocab_path, file)
                    data.append(keypoint_path)
        return data

    def create_label_to_index(self):
        labels = sorted(os.listdir(self.keypoints_dir))  # 키포인트 디렉토리의 모든 레이블 가져오기
        return {label: index for index, label in enumerate(labels)}  # 레이블을 인덱스에 매핑

    def load_keypoints(self, keypoint_path):
        return np.load(keypoint_path)

    def get_label_from_keypoint_path(self, keypoint_path):
        label = os.path.basename(os.path.dirname(keypoint_path))  # 파일 경로로부터 레이블 추출
        return self.label_to_index[label]  # 매핑된 인덱스를 반환

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoint_path = self.data[idx]
        keypoint_data = self.load_keypoints(keypoint_path)
        label = self.get_label_from_keypoint_path(keypoint_path)
        return keypoint_data, label
