from visdom import Visdom
import numpy as np
import torchvision
from PIL import ImageDraw, Image, ImageFont
import torch
import pdb


class VisdomPlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='gan'):
        self.viz = Visdom(port=7777)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='epoch'):
        x = list(range(y.shape[0]))
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.asarray([x]), Y=np.asarray([y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.asarray([x]), Y=np.asarray([y]), env=self.env, win=self.plots[var_name],
                                 name=split_name)

    def draw(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env)
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name])