import paddle
import paddle.vision.transforms as T
import config

transform = T.Compose(
    [
        T.ToTensor(),
    ]
)
import define_model
import matplotlib.pyplot as plt

#
model = define_model.Vae_paddle(in_len=784, latent_len=32, batch_size=32)
model_dict = paddle.load(f"{config.epochs}.model")
model.set_state_dict(model_dict)

mu_list = [0] * config.batch_size
sigmoid_list = [1] * config.batch_size
decode_input = model.sample(mu_list, sigmoid_list)
out = model.decode(decode_input)
out = out.reshape((config.batch_size, 28, 28))
for i in range(config.batch_size):
    temp = out[i, :, :]
    plt.imshow(temp.numpy())
    plt.show()
