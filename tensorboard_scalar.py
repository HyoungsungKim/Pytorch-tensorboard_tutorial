# * python tensorboard_scalar.py
# * tensorboard --logdir scalar --port=6006
import math
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter(logdir='scalar/tutorial')

    for step in range(-360, 360):
        angle_rad = step * math.pi / 180
        writer.add_scalar('sin', math.sin(angle_rad), step)
        writer.add_scalar('cos', math.cos(angle_rad), step)
        writer.add_scalars('sin and cos', {'sin': math.sin(angle_rad), 'cos': math.cos(angle_rad)}, step)
    writer.close()
