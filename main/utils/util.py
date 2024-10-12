import os
import cv2
import numpy as np
import torch

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def preprocess_video(video_frames):
    # 예: 비디오 프레임을 정규화하는 간단한 변환
    processed_frames = []
    for frame in video_frames:
        frame = cv2.resize(frame, (224, 224))  # 예: 크기 조정
        frame = frame / 255.0  # 정규화
        processed_frames.append(frame)
    return np.array(processed_frames)

def Vector_Normalization(joint):
    # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
    v = v2 - v1 
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

    angle = np.degrees(angle) # Convert radian to degree

    angle_label = np.array([angle], dtype=np.float32)

    return v, angle_label


def custom_collate_fn(batch):
    keypoints, labels = zip(*batch)
    
    # 모든 키포인트의 최대 길이 계산
    max_length = max([kp.shape[0] for kp in keypoints])

    # 각 키포인트를 패딩하여 동일한 길이로 만듦
    padded_keypoints = []
    for kp in keypoints:
        padding_length = max_length - kp.shape[0]
        if padding_length > 0:
            # 필요한 패딩만큼 (0, 0)으로 채운 더미 프레임 추가
            pad = torch.zeros((padding_length, kp.shape[1], kp.shape[2]))
            kp = torch.tensor(kp)  # numpy 배열을 Tensor로 변환
            kp = torch.cat((kp, pad), dim=0)
        else:
            kp = torch.tensor(kp)  # numpy 배열을 Tensor로 변환
        padded_keypoints.append(kp)

    stacked_keypoints = torch.stack(padded_keypoints)
    stacked_labels = torch.stack([torch.tensor(label) for label in labels])

    return stacked_keypoints, stacked_labels

