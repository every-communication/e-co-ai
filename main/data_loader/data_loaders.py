import numpy as np
import os
import json
import math
import tensorflow as tf
from tensorflow.python.keras.utils.all_utils import Sequence
from base.base_data_loader import BaseDataLoader
from tensorflow.python.data import Dataset

class KeyPointsDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, training=True, **kwargs):
        self.dataset = KeyPointDataset(data_dir=data_dir, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split)
    

class KeyPointDataset(Dataset):
    def __init__(self, data_dir, batch_size, num_samples, interval, keypoint_types, framework, mode='train', transforms=None):
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.interval = interval
        self.keypoint_types = keypoint_types
        self.framework = framework

        if transforms is None:
            self.transforms = self.default_transforms()
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        self._load_data()

    def default_transforms(self):
        return tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(1./255)
        ])

    def _load_data(self):
        print(self.data_dir)
        vocab_dirs = sorted(os.listdir(self.data_dir))
        for vocab in vocab_dirs:
            label = vocab
            if self.framework == 'mediapipe':
                for instance in os.listdir(os.path.join(self.data_dir, vocab)):
                    vocab_datas = os.path.join(self.data_dir, vocab, instance)
                    self.videos.append(vocab_datas)
                    self.labels.append(label)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path = self.videos[index]
        frame_keys = sorted(os.listdir(video_path))
        frame_keys = trim_action(video_path, frame_keys)
        
        frame_count = len(frame_keys)
        sampled_indices = self.sampling(frame_count, self.num_samples, self.interval)
        sampled_keys = [frame_keys[idx] for idx in sampled_indices]

        keypoints = []
        for frame_key in sampled_keys:
            with open(os.path.join(video_path, frame_key)) as json_file:
                data = json.load(json_file)

            total_keypoints = []
            for keypoint_type in self.keypoint_types:
                if self.framework == 'openpose':
                    type_keypoints = data["people"][f"{keypoint_type}_keypoints_2d"]
                    type_keypoints = reshape_keypoints(type_keypoints)
                else:  # mediapipe json
                    assert f"{keypoint_type}_keypoints" in data.keys(), f"{keypoint_type}_keypoints not exist in {video_path}/{frame_key}"
                    type_keypoints = np.array(data[f"{keypoint_type}_keypoints"])
                    type_keypoints = reshape_keypoints(type_keypoints)

                total_keypoints.append(type_keypoints)

            all_keypoints = np.concatenate(total_keypoints, axis=0)
            if self.format != 'image':
                all_keypoints = all_keypoints.flatten()
            if self.format == 'flatten':
                average_value = np.mean(all_keypoints)
                all_keypoints = average_value

            keypoints.append(all_keypoints)

        keypoints_data = np.stack(keypoints)  # Shape: [self.num_samples, 3 * num_keypoints]

        return keypoints_data, self.labels[index]

def trim_action(video_path, frame_keys):
    last_dir = os.path.basename(video_path)
    vocab = os.path.basename(os.path.dirname(video_path))
    morpheme_path = 'C:\\Users\\admin\\Documents\\github\\kslr\\dataset\\morpheme'

    # Process last_dir to extract relevant components
    if len(last_dir) > 1:
        _, _, vocab, collector, angle = last_dir.split('_')
        vocab = vocab[4:]
        collector = collector[4:]

    json_filepath = os.path.join(morpheme_path, vocab, collector, f'{angle}.json')

    if not os.path.exists(json_filepath):
        raise FileNotFoundError(f"No such file: '{json_filepath}'")

    with open(json_filepath, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        start = data["data"][0]["start"]
        end = data["data"][0]["end"]

    start_frame = math.floor(start * 30)
    end_frame = math.ceil(end * 30)

    trimmed = frame_keys[start_frame:end_frame]
    return trimmed

def reshape_keypoints(keypoints_data):
    x, y, z = split_coordinates(keypoints_data)
    normalized_data = merge_coordinates(normalize(x), normalize(y), normalize(z))
    reshaped_keypoints = tf.convert_to_tensor(normalized_data).reshape(-1, 3)
    return reshaped_keypoints

def split_coordinates(input_list):
    x_coordinates = input_list[::3]
    y_coordinates = input_list[1::3]
    z_coordinates = input_list[2::3]
    return x_coordinates, y_coordinates, z_coordinates

def merge_coordinates(x_values, y_values, z_values):
    merged_list = [coord for triplet in zip(x_values, y_values, z_values) for coord in triplet]
    return merged_list

def normalize(coordinates):
    c_array = np.array(coordinates)
    mean_value = np.mean(c_array)
    std_dev = np.std(c_array)
    normalized_coordinates = (c_array - mean_value) / std_dev
    return normalized_coordinates
