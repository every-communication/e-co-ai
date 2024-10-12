import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.p3d import P3D
from model.cnn_lstm import CNN_LSTM
from data_loader.data_loader import SignLanguageDataset
import argparse
from utils.util import custom_collate_fn
import numpy as np


SEED = 125
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(args):
    # 모든 vocab 폴더를 찾기
    vocab_names = [d for d in os.listdir(args.vocab_dir) if os.path.isdir(os.path.join(args.vocab_dir, d))]

    # 각 vocab에 대해 학습 루프
    for vocab in vocab_names:
        print(f"Training for vocab: {vocab}")

        # 데이터셋 및 DataLoader 준비
        train_dataset = SignLanguageDataset()  # data_dir로 수정
        print(f"Number of samples in the dataset: {len(train_dataset.data)}")

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

        # 모델 초기화
        if args.model == 'p3d':
            model = P3D(num_classes=70)  # 70개의 수어 클래스
        elif args.model == 'cnn_lstm':
            model = CNN_LSTM(num_classes=70)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 손실 함수 및 옵티마이저 설정
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 학습 루프
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            for keypoints, labels in train_loader:  # keypoints와 labels를 함께 unpack
                # keypoints와 labels를 torch.Tensor로 변환하고, device로 이동
                keypoints = torch.tensor(keypoints, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                if(args.model == 'p3d'):
                    keypoints = keypoints.unsqueeze(1)
                    keypoints = keypoints.repeat(1, 4, 1, 1, 1)
                elif args.model == 'cnn_lstm':
                    keypoints = keypoints.view(keypoints.size(0), keypoints.size(1), -1)

                optimizer.zero_grad()
                print(f'Input shape before model: {keypoints.shape}')
                outputs = model(keypoints)  # 모델에 키포인트 입력
                loss = criterion(outputs, labels)  # labels 사용
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader)}")

        print(f"Training complete for vocab: {vocab}\n")

    print("All vocab training complete.")

if __name__ == "__main__":
    # 인자 파서 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['p3d', 'cnn_lstm'], required=True, help="Choose model: 'p3d' or 'cnn_lstm'")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--vocab_dir', type=str, default='dataset/processed_video', help='Directory containing vocab folders')
    args = parser.parse_args()

    main(args)
