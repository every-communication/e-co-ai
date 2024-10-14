import numpy as np
import os
import json
import math
import tensorflow as tf

class KeyPointsDataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']
        self.shuffle = config.get('shuffle', True)
        
        self.dataset = KeyPointDataset(config)
        self.train_data, self.valid_data = self._split_dataset()

    def _split_dataset(self):
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        np.random.seed(0)
        np.random.shuffle(indices)

        if isinstance(self.validation_split, float):
            len_valid = int(n_samples * self.validation_split)
        else:
            len_valid = self.validation_split
        
        train_indices, valid_indices = indices[len_valid:], indices[:len_valid]
        train_data = tf.data.Dataset.from_tensor_slices(train_indices).map(self.dataset.__getitem__)
        valid_data = tf.data.Dataset.from_tensor_slices(valid_indices).map(self.dataset.__getitem__)

        return train_data.batch(self.batch_size), valid_data.batch(self.batch_size)

    def get_train_data(self):
        if self.shuffle:
            return self.train_data.shuffle(buffer_size=len(self.dataset))
        return self.train_data

    def get_validation_data(self):
        return self.valid_data


class KeyPointDataset:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        
        self.batch_size = config['batch_size']
        self.num_samples = config['frames']
        self.interval = config['interval']
        self.keypoint_types = config['keypoints_types']
        self.format = config.get('format', 'flatten')
        self.framework = config.get('framework', 'mediapipe')

        self.transforms = self.default_transforms()
        
        self.keypoints = [] # file name 들어가 있음
        self.labels = [] # vocab 들어가 있음
        self._load_data()

    def default_transforms(self):
        return tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(1./255)
        ])

    def _load_data(self):
        vocab_dirs = sorted(os.listdir(self.data_dir))
        for vocab in vocab_dirs:
            label = vocab
            for instance in os.listdir(os.path.join(self.data_dir, vocab)):
                vocab_data = os.path.join(self.data_dir, vocab, instance)
                self.keypoints.append(vocab_data)
                self.labels.append(label)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        index = tf.py_function(func=lambda x: x.numpy(), inp=[index], Tout=tf.int64)
    
        video_path = self.keypoints[index]
        frame_keys = sorted(os.listdir(video_path))
        frame_keys = self.trim_action(video_path, frame_keys)
        
        sampled_keys = self.sample_frames(frame_keys)
        keypoints_data = self.load_keypoints(video_path, sampled_keys)
        
        return keypoints_data, self.labels[index]

    def sample_frames(self, frame_keys):
        frame_count = len(frame_keys)
        sampled_indices = np.linspace(0, frame_count - 1, self.num_samples, dtype=int)
        return [frame_keys[idx] for idx in sampled_indices]

    def load_keypoints(self, video_path, frame_keys):
        keypoints = []
        for frame_key in frame_keys:
            with open(os.path.join(video_path, frame_key)) as json_file:
                data = json.load(json_file)
            total_keypoints = []

            for keypoint_type in self.keypoint_types:
                if self.framework == 'openpose':
                    type_keypoints = data["people"][f"{keypoint_type}_keypoints_2d"]
                else:  # mediapipe json
                    type_keypoints = data[f"{keypoint_type}_keypoints"]
                type_keypoints = self.reshape_keypoints(type_keypoints)
                total_keypoints.append(type_keypoints)

            keypoints.append(np.concatenate(total_keypoints, axis=0).flatten())
        
        return np.stack(keypoints)

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
