"""
Module to train DBN on MNIST.
"""

from os import path

from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from theano import config

#config.exception_verbosity='high'
#config.optimizer='fast_compile'
#config.mode='DEBUG_MODE'

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path, nvis, hidden):
    yaml = open(yaml_file, "r").read()
    input_dim = 784 # MNIST input size
    hyperparams = {"nvis": nvis,
                    "batch_size": 50,
                    "detector_layer_dim": hidden,
                    "monitoring_batches": 10,
                    "train_stop": 50000,
                    "max_epochs": 300,
                    "save_path": save_path
                  }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_rbm():
    hiddens = [200, 200, 200]
    yamls = ["rbm.yaml", "dbn1.yaml", "dbn2.yaml"]
    retrains = [False, False, True]
    for i, (hidden, yaml, retrain) in enumerate(zip(hiddens, yamls, retrains)):
        yaml_file = path.join(path.abspath(path.dirname(__file__)), yaml)
        if not path.isfile(yaml_file):
            retrain = True
        if not retrain:
            continue
        if i == 0:
            nvis = 784
        else:
            nvis = hiddens[i-1]
        save_path = path.abspath(path.dirname(__file__))
        train(yaml_file, save_path, nvis, hidden)

if __name__ == "__main__":
    train_rbm()