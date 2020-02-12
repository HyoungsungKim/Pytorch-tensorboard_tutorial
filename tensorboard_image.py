import torch
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

datasets_dir = './data'
composed = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=datasets_dir, train=True, download=True, transform=composed)
# validation_dataset = datasets.MNIST(root=datasets_dir, train=False, download=True)

image_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1)

if __name__ == '__main__':
    writer = SummaryWriter(logdir='images/MNIST')

    # * Image is (channel, y, x)
    for count, (image, tag) in enumerate(image_loader):
        writer.add_image('MNIST Tag is ' + str(tag), image[0])
        if count >= 3:
            break

    # * Image is (number, channel, y, x)
    for count, (image, tag) in enumerate(image_loader):
        writer.add_images('MNIST group', image)
        if count >= 3:
            break

    writer.close()
