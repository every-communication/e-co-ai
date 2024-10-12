import tensorflow as tf
from tensorflow.python.keras import Model, optimizers, losses
from logger import TensorboardWriter  # TensorboardWriter를 TensorFlow에 맞게 수정해야 할 수 있습니다.
import os

class BaseTrainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.tflite_dir = cfg_trainer['tflite_dir']

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = float('inf') if self.mnt_mode == 'min' else -float('inf')
            self.early_stop = cfg_trainer.get('early_stop', float('inf'))
            if self.early_stop <= 0:
                self.early_stop = float('inf')

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # 로그 기록
            log = {'epoch': epoch}
            log.update(result)

            # 로그 출력
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 모델 성능 평가
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        # 최상의 모델을 TFLite로 변환
        best_model_path = os.path.join(self.checkpoint_dir, 'model_best.h5')
        self.logger.info("변환 중: model_best.h5를 TFLite 형식으로 변환 중입니다.")
        self._convert_to_tflite(best_model_path)

    def _save_checkpoint(self, epoch, save_best=False):
        model_path = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_{}.h5'.format(epoch))
        self.model.save(model_path)
        self.logger.info("체크포인트 저장: {} ...".format(model_path))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.h5')
            self.model.save(best_path)
            self.logger.info("최고 모델 저장: model_best.h5 ...")

    def _convert_to_tflite(self, model_path):
        # TFLite 모델 변환
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(self.tflite_dir, 'model_best.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        self.logger.info(f"TFLite 모델 저장 완료: {tflite_path}")