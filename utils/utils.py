import argparse
from matplotlib import pyplot as plt

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def save_loss_plot_train_val(loss1, loss2, title, legend, path):
    plt.figure()
    plt.plot([i for i in range(len(loss1))], loss1)
    plt.plot([i for i in range(len(loss2))], loss2)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(path + '.png')