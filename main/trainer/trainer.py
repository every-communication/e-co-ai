import tensorflow as tf
from base.base_trainer import BaseTrainer
from tensorflow.python.keras import Model, optimizers, losses
from logger import TensorboardWriter  # TensorboardWriter를 TensorFlow에 맞게 수정해야 할 수 있습니다.
import os

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            with tf.GradientTape() as tape:
                output = self.model(data)
                loss = self.criterion(target, output)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            total_loss += loss.numpy()  # NumPy 배열로 변환

            # 로그 기록 및 TensorBoard 업데이트
            self.writer.set_step(epoch * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss, global_step=batch_idx)

        log = {'loss': total_loss / len(self.data_loader)}
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.evaluate(self.valid_data_loader)
        return {'val_loss': val_loss}  # validation loss를 반환하는 방식에 맞춰 수정 필요