import numpy as np
import os
import json
import tensorflow as tf

class KeyPointsDataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']
        self.shuffle = config.get('shuffle', True)
        
        self.dataset = KeyPointDataset(config)

    def _split_dataset(self):
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        np.random.seed(0)
        np.random.shuffle(indices)

        len_valid = int(n_samples * self.validation_split)
        train_indices, valid_indices = indices[len_valid:], indices[:len_valid]
        
        
        # n_sample = file 개수 ( .npy )
        # 해야 할 것 . 각 파일의 안에 튜플을 불러오기  
        # 한 파일에는 영상이기 때문에 프레임 당 left_hand, right_hand, face_bbox가 dict 형태로 저장되어 있는 듯?
        print("dataset")
        print("length : ", len(self.dataset))
        # Tensor 인덱스를 정수로 변환하여 __getitem__에 전달하는 함수
        print(self.dataset[0])
        print("data is : ", self.dataset[0][0])
        print("label is : ", self.dataset[0][1])

        # train_data와 valid_data를 리스트로 만들고 numpy 배열로 변환
        train_data = [self.dataset[i] for i in train_indices]
        valid_data = [self.dataset[i] for i in valid_indices]

        # 각 데이터의 모양을 출력하여 확인
        for i, data in enumerate(train_data):
            keypoints_data, label = data  # 튜플 언패킹
            print(f"Train Data {i} Keypoints Shape: {keypoints_data.shape}, Label: {label}")
        for i, data in enumerate(valid_data):
            keypoints_data, label = data  # 튜플 언패킹
            print(f"Valid Data {i} Keypoints Shape: {keypoints_data.shape}, Label: {label}")


        # np.array로 변환
        train_data = np.array(train_data)
        valid_data = np.array(valid_data)


        return train_data.batch(self.batch_size), valid_data.batch(self.batch_size)
    
    def load_data(self):
        return self._split_dataset()

    def get_train_data(self):
        if hasattr(self.dataset, 'cardinality'):
            buffer_size = self.dataset.cardinality().numpy()
        else:
            buffer_size = len(self.dataset)  # 이 경우 len(self.dataset)이 정수값인지 확인

        # buffer_size가 0 이하가 되지 않도록 설정
        buffer_size = max(1, buffer_size)
        
        return self.train_data.shuffle(buffer_size=buffer_size) if self.shuffle else self.train_data

    def get_validation_data(self):
        return self.valid_data


class KeyPointDataset:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        self.keypoints = [] # .npy 파일 경로 목록
        self.labels = [] # vocab (label) 목록
        self._load_data()

    def _load_data(self):
        vocab_dirs = sorted(os.listdir(self.data_dir))
        for vocab in vocab_dirs:
            label = vocab
            for instance in os.listdir(os.path.join(self.data_dir, vocab)):
                vocab_data = os.path.join(self.data_dir, vocab, instance)
                self.keypoints.append(vocab_data)
                self.labels.append(label)
                self.load_keypoints(os.path.join(self.data_dir, vocab, instance))

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        index = int(index)
        keypoints_path = self.keypoints[index]
        keypoints_data = self.load_keypoints(keypoints_path)

        return keypoints_data, self.labels[index]
    
    def load_keypoints(self, keypoints_path):
        keypoints = []
        data = np.load(keypoints_path, allow_pickle=True)
        print("keypoints data loader - load_keypoints - data shape", keypoints_path, " is ", data.shape)
        if data.size > 0:
            keypoints.append(data)
        if keypoints:
            return np.stack(keypoints)
        else:
            print("No data to stack for:", keypoints_path)
            return None

    def reshape_keypoints(self, keypoints_data):
        x, y, z = self.split_coordinates(keypoints_data)
        normalized_data = self.merge_coordinates(self.normalize(x), self.normalize(y), self.normalize(z))
        return np.array(normalized_data).reshape(-1, 3)

    def split_coordinates(self, input_list):
        return input_list[::3], input_list[1::3], input_list[2::3]

    def merge_coordinates(self, x_values, y_values, z_values):
        return [coord for triplet in zip(x_values, y_values, z_values) for coord in triplet]

    def normalize(self, coordinates):
        c_array = np.array(coordinates)
        return (c_array - np.mean(c_array)) / np.std(c_array)
