import tensorflow as tf
from base.base_trainer import BaseTrainer
from data_loader.data_loaders import KeyPointsDataLoader
from tensorflow.python.keras import Model, optimizers, losses
from logger import TensorboardWriter  # TensorboardWriter를 TensorFlow에 맞게 수정해야 할 수 있습니다.
import os

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader):
        self.device = device
        self.data_loader = data_loader
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        

    def _train_epoch(self, epoch):
        total_loss = 0
        print(self.data_loader)
        print('\n')
        dataset = self.data_loader.get_train_data()  # get_train_data() 호출

        for batch_idx in range(len(dataset) // self.data_loader.batch_size):
            # 배치 데이터 가져오기
            data, target = dataset[batch_idx]
            data, target = data.to(self.device), target.to(self.device)

            with tf.GradientTape() as tape:
                output = self.model(data)
                loss = self.criterion(target, output)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            total_loss += loss.numpy()  # NumPy 배열로 변환

            # 로그 기록 및 TensorBoard 업데이트
            self.writer.set_step(epoch * (len(dataset) // self.data_loader.batch_size) + batch_idx)
            self.writer.add_scalar('loss', loss.numpy(), global_step=batch_idx)

        log = {'loss': total_loss / (len(dataset) // self.data_loader.batch_size)}
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log 

    def _valid_epoch(self, epoch):
        # Validation 로직 추가
        val_loss = 0
        # Validation 데이터 로더를 사용해 평가
        for batch_idx in range(len(self.valid_data_loader)):  # valid_data_loader의 길이만큼 반복
            data, target = self.valid_data_loader[batch_idx]
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(target, output)
            val_loss += loss.numpy()  # validation loss 누적

        return {'val_loss': val_loss / len(self.valid_data_loader)}  # 평균 validation loss 반환
