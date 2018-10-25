__author__ = "Capri Lee"
__copyright__ = "Capri Lee 2018"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import json
import time
import math
# import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        ndim=configs['model']['layers'][0]['input_dim'],
        normalise=configs['data']['normalise']
    )

    # in-memory training
    # model.train(
    # 	x,
    # 	y,
    # 	epochs = configs['training']['epochs'],
    # 	batch_size = configs['training']['batch_size']
    # )

    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            ndim=configs['model']['layers'][0]['input_dim'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir = configs['model']['save_dir']
    )

    # x_test, y_test = data.get_test_data(
    # 	seq_len = configs['data']['sequence_length'],
    # 	ndim = configs['model']['layers'][0]['input_dim'],
    # 	normalise = configs['data']['normalise']
    # )

    predictions = model.predict_sequences_multiple(x, configs['data']['sequence_length'],configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x)
    print(predictions[-1])

# plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
# plot_results(predictions, y_test)

if __name__ == '__main__':
    main()
