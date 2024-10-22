import tensorflow as tf
from data_loader.data_loader_factory import KeyPointsDataLoader
from tensorflow.python.keras import Model, optimizers, losses
from sklearn.preprocessing import LabelEncoder
from logger import TensorboardWriter  # TensorboardWriter를 TensorFlow에 맞게 수정해야 할 수 있습니다.
import os

class Trainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader):
        self.device = device
        self.data_loader = data_loader
        
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

        self.do_validation = True
        
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
        self.model.save_weights(model_path)
        self.logger.info("체크포인트 저장: {} ...".format(model_path))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.h5')
            self.model.save_weights(best_path)
            self.logger.info("최고 모델 저장: model_best.h5 ...")

    def _convert_to_tflite(self, model_path):
        # TFLite 모델 변환
        self.model.load_weights(model_path)
        # TFLite 변환기 설정
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model) # best_model은 훈련된 Keras 모델입니다.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert() # TensorList 관련 설정
        tflite_path = os.path.join(self.tflite_dir, 'model_best.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        self.logger.info(f"TFLite 모델 저장 완료: {tflite_path}")

    def _train_epoch(self, epoch):
        total_loss = 0
        
        train_data, valid_data = self.data_loader.load_data()
        # 배치 shape 확인
        for batch in train_data.take(1):
            print("Sample batch shape: ", batch[0].shape)  # 데이터 shape 확인
            break

        for batch_idx, (data, label) in enumerate(train_data):
            with tf.GradientTape() as tape:
                output = self.model(data)  # 모델에 데이터를 입력하여 예측값 생성
                output = output - 1
                label = label - 1
                loss = self.criterion(label, output)  # 손실 계산

            # Gradient 계산 및 적용
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            total_loss += loss.numpy()  # 손실값을 NumPy로 변환하여 더하기

            # 로그 기록 및 TensorBoard 업데이트
            self.writer.set_step(epoch * len(train_data) + batch_idx)
            self.writer.add_scalar('loss', loss.numpy())
        
        # 에포크 평균 손실값 계산
        log = {'loss': total_loss / len(train_data)}
        
        # 검증 수행 (선택적)
        if self.do_validation:
            val_log = self._valid_epoch(epoch, valid_data)
            log.update(val_log)

        # 학습률 조정 (선택적)
        #if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch, valid_data):
        # Validation 로직 추가
        val_loss = 0
        
        # Validation 데이터 로더를 사용해 평가
        for batch_idx, (data, label) in enumerate(valid_data):
            output = self.model(data)
            output = output - 1
            label = label - 1
            loss = self.criterion(label, output)
            val_loss += loss.numpy()

        return {'val_loss': val_loss / len(valid_data)}  # 평균 validation loss 반환
