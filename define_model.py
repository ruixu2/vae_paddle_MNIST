import paddle
from paddle import nn
class Vae_paddle(paddle.nn.Layer):
    def __init__(self, in_len, latent_len, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.latent_len = latent_len
        self.encode_in = nn.Linear(in_len, 512)
        self.encode_l = nn.Linear(512, 256)
        self.encode_out = nn.Linear(256, 2 * latent_len)

        self.decode_in = nn.Linear(self.latent_len, 128)
        self.decode_l = nn.Linear(128, 256)
        self.decode_out = nn.Linear(256, in_len)

        self.a = nn.Tanh()
        self.a2=nn.Sigmoid()

    def encode(self, x):
        x = self.encode_in(x)
        x = self.a(x)
        x = self.encode_l(x)
        x = self.a2(x)
        x = self.encode_out(x)
        x = self.a(x)
        return x[:, :self.latent_len], x[:, self.latent_len:]

    def sample(self, mu, sigmod):
        x = []
        for mean, std in zip(mu, sigmod):
            x.append(paddle.normal(mean, std, shape=(self.batch_size, 1)))
        res = paddle.concat(x)
        res = res.reshape((self.batch_size, self.latent_len))
        # print(res.shape)
        return res

    def decode(self, x):
        x = self.decode_in(x)
        x = self.a(x)
        x = self.decode_l(x)
        x = self.a2(x)
        x = self.decode_out(x)
        x = self.a(x)
        return x

    def forward(self, x):
        mu, sigmoid = self.encode(x)
        decode_input = self.sample(mu, sigmoid)
        out = self.decode(decode_input)
        return mu, sigmoid, out