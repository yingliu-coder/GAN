import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
from gan import discrimination, generate
import torch.nn as nn

batch_sz = 100
device = torch.device('cuda')

train_data = dataloader.DataLoader(datasets.MNIST(root='data/', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]), download=True), shuffle=True, batch_size=batch_sz)

num_epoch = 100
num_test_samples = 3
num_batches = len(train_data)

def images2vectors(images):
    return images.view(images.size(0), 784)

def vectors2images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    n = torch.randn(size, 100)
    return n.to(device)

discriminator = discrimination().to(device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

generator = generate().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

loss_fn = nn.BCELoss()

def train_discriminator(optimizer, loss_fn, real_data, fake_data):
    optimizer.zero_grad()

    discriminator_real_data = discriminator(real_data)
    loss_real = loss_fn(discriminator_real_data, torch.ones(real_data.size(0), 1).to(device))
    loss_real.backward()

    discriminator_fake_data = discriminator(fake_data)
    loss_fake = loss_fn(discriminator_fake_data, torch.zeros(fake_data.size(0), 1).to(device))
    loss_fake.backward()

    optimizer.step()

    return loss_real + loss_fake, discriminator_real_data, discriminator_fake_data

def train_generator(optimizer, loss_fn, fake_data):
    optimizer.zero_grad()

    output_discriminator = discriminator(fake_data)
    loss = loss_fn(output_discriminator, torch.ones(output_discriminator.size(0), 1).to(device))
    loss.backward()
    optimizer.step()
    return loss

plt.figure()

for epoch in range(num_epoch):

    for train_idx, (input_real_batch, _) in enumerate(train_data):
        real_data = images2vectors(input_real_batch).to(device)
        generated_fake_data = generator(noise(real_data.size(0))).detach()
        d_loss, discriminated_real, discriminated_fake = train_discriminator(d_optimizer, loss_fn, real_data,
                                                                             generated_fake_data)

        generated_fake_data = generator(noise(real_data.size(0)))
        g_loss = train_generator(g_optimizer, loss_fn, generated_fake_data)

        if train_idx == len(train_data) - 1:
            print(epoch, 'd_loss: ', d_loss.item(), 'g_loss: ', g_loss.item())

    test_noise = noise(num_test_samples)
    generated = generator(test_noise).detach()
    discriminated = discriminator(generated)
    images = vectors2images(generated).cpu().numpy()
    plt.subplot(10, 10, epoch + 1)
    plt.imshow(images[0][0], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])

plt.show()


print('End Training')