import paddle.nn
from paddle.vision.datasets import MNIST
from paddle import nn
import paddle.vision.transforms as T
import numpy as np

import config
import define_model

transform = T.Compose(
    [
        T.ToTensor(),
    ]
)
train_dataset = MNIST(mode='train', transform=transform)
test_dataset = MNIST(mode='test', transform=transform)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

model = define_model.Vae_paddle(in_len=784, latent_len=32, batch_size=32)
opt = paddle.optimizer.Adam(parameters=model.parameters())
# print(model.s)
for epoch in range(1,config.epochs):
    print(f"{epoch}/{config.epochs}")
    train_epoch_loss = []
    test_epoch_loss = []
    for idx, (x, label) in enumerate(train_loader):
        x = x.reshape((config.batch_size, 28 * 28))
        mu, sigmoid, y = model(x)
        # print(y.shape)
        loss1 = nn.functional.mse_loss(x, y)
        loss2 = 0.5 * (-paddle.log(sigmoid * sigmoid) + mu * mu + sigmoid * sigmoid - 1)
        loss = loss1 + 0.1 * loss2
        train_epoch_loss.append(loss.numpy())
        opt.clear_grad()
        loss.backward()
        # print(loss.numpy())
        opt.step()
        # print(loss)
    print(f"train loss: {np.mean(train_epoch_loss)}")
    for idx, (x, label) in enumerate(test_loader):
        x = x.reshape((config.batch_size, 28 * 28))
        mu, sigmoid, y = model(x)
        # print(y.shape)
        loss1 = nn.functional.mse_loss(x, y)
        loss2 = 0.5 * (-paddle.log(sigmoid * sigmoid) + mu * mu + sigmoid * sigmoid - 1)
        loss = loss1 + 0.1 * loss2
        test_epoch_loss.append(loss.numpy())
    print(f"test loss: {np.mean(test_epoch_loss)}")
paddle.save(model.state_dict(), f"{config.epochs}.model")
