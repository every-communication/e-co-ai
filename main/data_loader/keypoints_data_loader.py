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
        len_valid -= (len_valid % 10)
        train_indices, valid_indices = indices[len_valid:], indices[:len_valid]
        
        # n_sample = file 개수 ( .npy )
        #print("dataset")
        print("length : ", len(self.dataset))
        
        # train_data와 valid_data를 리스트로 만들고 numpy 배열로 변환
        #print("dataset[0] length : ", len(self.dataset[0])) # (keypoints, label)
        
        loaded_train_data = [self.dataset[i] for i in train_indices]
        loaded_valid_data = [self.dataset[i] for i in valid_indices]

        train_data = [loaded_train_data[i][0] for i in range(len(loaded_train_data))]
        valid_data = [loaded_valid_data[i][0] for i in range(len(loaded_valid_data))]
        train_label = [loaded_train_data[i][1] for i in range(len(loaded_train_data))]
        valid_label = [loaded_valid_data[i][1] for i in range(len(loaded_valid_data))]
        
        # numpy 배열로 변환
        train_data = np.array(train_data)
        valid_data = np.array(valid_data)

        # TensorFlow Dataset 생성 및 배치 나누기
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data, valid_label)).batch(self.batch_size)

        # 배치 확인
        #for batch in train_dataset:
            #print(batch.shape)   # BatchDataSet(batch_size, frames, keypoints, 2d-coordinate) (10, 256, 44, 2)
        #for batch in valid_dataset:
            #print(batch.shape)
        
        return train_dataset, valid_dataset
    
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
        self.frame_length = config['frames']
        self.keypoints = [] # .npy 파일 경로 목록
        self.labels = [] # vocab (label) 목록
    
        self._load_data()

    def _load_data(self):
        print("_load_data")
        vocab_dirs = sorted(os.listdir(self.data_dir))
        for vocab in vocab_dirs:
            label = vocab
            for instance in os.listdir(os.path.join(self.data_dir, vocab)):
                vocab_data = os.path.join(self.data_dir, vocab, instance)
                self.keypoints.append(vocab_data)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        index = int(index)
        keypoints_path = self.keypoints[index]
        keypoints_data = self.load_keypoints(keypoints_path)

        return keypoints_data, self.labels[index]
    
    def load_keypoints(self, keypoints_path):
        data = np.load(keypoints_path, allow_pickle=True)
        # 데이터의 프레임을 256으로 맞추기
        if data.shape[0] < self.frame_length:
            # 데이터가 256보다 작을 경우 패딩
            pad_width = self.frame_length - data.shape[0]
            # data와 동일한 차원의 0으로 채운 배열 생성
            padding = np.zeros((pad_width, *data.shape[1:]))  # 패딩의 차원 조정
            data = np.vstack((data, padding))
        elif data.shape[0] > self.frame_length:
            # 데이터가 256보다 클 경우 자름
            data =  data[:self.frame_length]

        return data