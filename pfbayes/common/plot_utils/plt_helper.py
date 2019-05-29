from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from matplotlib import cm


def plot_image_seqs(image_seqs, save_prefix):
    for t in range(len(image_seqs)):
        num_imgs = len(image_seqs[t])
        for i in range(num_imgs):
            plt.subplot(1, num_imgs, i + 1)
            plt.imshow(image_seqs[t][i])
            plt.axis('equal')
            plt.axis('off')
        output_path = os.path.join(save_prefix + '-%05d.png' % t)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


def create_video(save_prefix, output_name, framerate=2):
    bashCommand = 'ffmpeg -framerate {} -y -i {} {}'.format(framerate, os.path.join(save_prefix + '-%05d.png'), output_name)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


colors = dict()
colors['MPF'] = cm.Paired(5)
colors['Bootstrap'] = cm.Paired(3)
colors['KBR'] = cm.Paired(9)

colors['One-Pass SMC'] = cm.Paired(3)
colors['SMC'] = cm.Paired(3)
colors['PMD'] = cm.Paired(7)

colors['SVI'] = cm.Paired(1)
colors['SGD NPV'] = cm.Paired(9)
colors['SGD Langevin'] = cm.Paired(11)

