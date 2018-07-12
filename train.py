'''
    main file for start the server that capture the info from the mic and then
    spot the keywords
'''

import json
import argparse
import numpy as np

import models.conv_net as cnn

FLAG = None

def main():
    print('Loading configuration file...')

    if not FLAG.config:
        FLAG.config = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            "config.json")
    with open(FLAG.config) as f:
        config = json.loads(f.read())

    model = cnn.convolutional_network_2_layer(31, (99,40,1), learning_rate = config['model']['learning_rate'])
    model = cnn.train_model("/nfsd/hda/vaninedoar/HDA-Project/compressed_feature", model, 1, config['model']['epoch'], True, "saved_model", batch_size = config['model']['batch_size'])

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
