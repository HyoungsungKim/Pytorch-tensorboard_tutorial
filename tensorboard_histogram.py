from tensorboardX import SummaryWriter
import torch

import numpy as np

if __name__ == '__main__':
    writer = SummaryWriter(logdir='histogram/hist')
    sigma = 1
    for step in range(5):
        writer.add_histogram('hist-numpy', np.random.normal(0, sigma, 1000), step)
        sigma += 1

    sigma = 1
    for step in range(5):
        torch_normal = torch.distributions.Normal(0, sigma)
        writer.add_histogram('hist-torch', torch_normal.sample((1, 1000)), step)
        sigma += 1

    writer.close()
