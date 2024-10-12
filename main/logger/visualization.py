import importlib
from datetime import datetime
import tensorflow as tf

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None

        if enabled:
            log_dir = str(log_dir)
            self.writer = tf.summary.create_file_writer(log_dir)

            if self.writer is None:
                message = "Warning: visualization (TensorBoard) is configured to use, but it could not be initialized."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        # 기록하기 위한 함수 이름들?
        self.tb_writer_ftns = {
            'scalar', 'scalars', 'image', 'images', 'audio',
            'text', 'histogram', 'pr_curve', 'embedding'
        }
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def add_scalar(self, tag, data, step=None):
        if step is None:
            step = self.step
        with self.writer.as_default():
            tf.summary.scalar(f'{tag}/{self.mode}', data, step=step)
            self.writer.flush()

    def add_scalars(self, tag, data, step=None):
        if step is None:
            step = self.step
        with self.writer.as_default():
            for key, value in data.items():
                tf.summary.scalar(f'{tag}/{key}/{self.mode}', value, step=step)
            self.writer.flush()

    def add_image(self, tag, image, step=None):
        if step is None:
            step = self.step
        with self.writer.as_default():
            tf.summary.image(f'{tag}/{self.mode}', image, step=step)
            self.writer.flush()

    # You can implement additional methods for images, audio, histograms, etc., similarly
    # Just follow the pattern established with the add_scalar and add_scalars methods.

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            raise AttributeError.
        """
        if name in self.tb_writer_ftns:
            raise AttributeError(f"Method '{name}' is not implemented for TensorBoard in this context.")
        else:
            # Default action for returning methods defined in this class, set_step() for instance.
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
