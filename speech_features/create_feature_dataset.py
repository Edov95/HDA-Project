import json
import argparse
import loadMFCC as lw
import os
import numpy as np

FLAG = None

def main():

    print('Loading configuration file...')

    if not FLAG.config:
        FLAG.config = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            "config.json")
    with open(FLAG.config) as f:
        config = json.loads(f.read())

    print('Configuration file loaded')

    print('Loading training dataset...')
    x_train, y_train = lw.load_dataset(config['words_training']['data_dir'],
        config['words_training']['wanted_words'],
        config['words_training']['noise_percentage'], 'training',
        config['words_training']['noise_volume_range'])
    print('Trainig dataset loaded')
    print('Loading validation dataset...')
    x_validation, y_validation = lw.load_dataset(config['words_training']['data_dir'],
        config['words_training']['wanted_words'],
        config['words_training']['noise_percentage'], 'validation',
        config['words_training']['noise_volume_range'])
    print('Validation dataset loaded')
    print('Loading test dataset...')
    x_test, y_test = lw.load_dataset(config['words_training']['data_dir'],
        config['words_training']['wanted_words'],
        config['words_training']['noise_percentage'], 'testing',
        config['words_training']['noise_volume_range'])
    print('Test dataset loaded')

    np.save(os.path.join(config['words_training']['saved_path'], 'x_trainMFCC.npy'), x_train)
    print('x_train saved')
    np.save(os.path.join(config['words_training']['saved_path'], 'y_trainMFCC.npy'), y_train)
    print('y_train saved')
    np.save(os.path.join(config['words_training']['saved_path'], 'x_validationMFCC.npy'), x_validation)
    print('x_validation saved')
    np.save(os.path.join(config['words_training']['saved_path'], 'y_validationMFCC.npy'), y_validation)
    print('y_validation saved')
    np.save(os.path.join(config['words_training']['saved_path'], 'x_testMFCC.npy'), x_test)
    print('x_test saved')
    np.save(os.path.join(config['words_training']['saved_path'], 'y_testMFCC.npy'), x_test)
    print('y_test saved')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type = str,
        default = "config.json",
        help = "The config file to use"
    )

    FLAG, _ = parser.parse_known_args()
    main()
