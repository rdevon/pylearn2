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

def train(yaml_file, save_path, nvis, hidden, pretrain=None):
    yaml = open(yaml_file, "r").read()
    input_dim = 784 # MNIST input size
    hyperparams = {"nvis": nvis,
                    "batch_size": 50,
                    "detector_layer_dim": hidden,
                    "monitoring_batches": 10,
                    "train_stop": 50000,
                    "max_epochs": 10,
                    "save_path": save_path,
                    "pretrain": pretrain
                  }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_rbm():
    hiddens = [200, 200, 200, 10]
    names = ["rbm", "dbn1", "dbn2", "mlp"]
    retrains = [False, False, True, True]
    p = path.abspath(path.dirname(__file__))
    for i, (hidden, name, retrain) in enumerate(zip(hiddens, names, retrains)):
        yaml_file = path.join(p, name + ".yaml")
        pkl_file = path.join(p, name + ".pkl")
        if not path.isfile(pkl_file):
            retrain = True

        if not retrain:
            continue
        if i == 0 or name == "mlp":
            nvis = 784
        else:
            nvis = hiddens[i-1]
        if i > 0:
            pretrain = path.join(p, names[i-1] + ".pkl")
        else:
            pretrain = None
        save_path = p
        train(yaml_file, save_path, nvis, hidden, pretrain)

if __name__ == "__main__":
    train_rbm()