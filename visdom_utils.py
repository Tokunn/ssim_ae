from visdom import Visdom

import numpy as np
import matplotlib.pyplot as plt
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, pngpath, env_name='main', host_name='localhost'):
        self.viz = Visdom(host_name)
        self.env = env_name
        self.plots = {}
        self.saveplots = {}
        self.pngpath = pngpath

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

        # Matplotlib
        if var_name+split_name not in self.saveplots:
            self.saveplots[var_name+split_name] = ([x], [y])
        else:
            self.saveplots[var_name+split_name][0].append(x)
            self.saveplots[var_name+split_name][1].append(y)

        plt.figure()
        for name in self.saveplots.keys():
            plt.plot(self.saveplots[name][0], self.saveplots[name][1])
        plt.savefig(os.path.join(self.pngpath, name+'.png'))
