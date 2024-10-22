import argparse
import collections
import numpy as np
import tensorflow as tf
import keras.src.optimizers as optimizers
from data_loader.data_loader_factory import DataLoaderFactory
from model.model_factory import ModelFactory
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from config.config_parser import ConfigParser
from trainer import Trainer
import json

# fix random seeds for reproducibility
SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)

def main(config):
    # config == ConfigParser
    logger = config.get_logger('train') #TODO NOT LOGGER FILE

    # setup data_loader instances
    data_loader = DataLoaderFactory.get_data_loader(config['data_loader']['type'], dict(config['data_loader']['args']))

    # build model architecture, then print to console
    model = ModelFactory.get_model(config['arch']['type'], dict(config['arch']['args']))          
    logger.info(model) #TODO logger

    # prepare for (multi-device) GPU training
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    logger.info(f"Using device: {device}")

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']] #TODO metric

    # build optimizer
    #optimizer = config.init_obj('optimizer', optimizers)
    optimizer = optimizers.Adam(learning_rate=config['optimizer']['args']['learning_rate'])

    trainer = Trainer(model, criterion, metrics, optimizer, config, device, data_loader)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Tensorflow Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;learning_rate'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs')
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
