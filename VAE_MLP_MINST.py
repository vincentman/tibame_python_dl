import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import objectives

# 讀取MNIST 數據
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_valid = X_valid.astype('float32') / 255.

original_dim = 784
X_train_flatten = np.reshape(X_train, [-1, original_dim])
X_valid_flatten = np.reshape(X_valid, [-1, original_dim])

print('X_train_flatten.shape = ', X_train_flatten.shape, ', y_train.shape = ', y_train.shape)
print('X_valid_flatten.shape = ', X_valid_flatten.shape, ', y_valid.shape = ', y_valid.shape)

# 設定網路參數
batch_size = 16
epochs = 50
latent_dim = 2
intermediate_dim = 256
epsilon_std = 1.0

# 建立 Encoder
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


# 取樣函數
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# 建立 Decoder
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')
decoder_mean = Dense(original_dim, activation='sigmoid', name='decoder_mean')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# LOSS
# modified calculation of loss
def vae_loss(x, x_decoded_mean):
    # xent_loss is reconstruction error
    # use binary_crossentropy because this is not multi-label classification or regression but unsupervised problem
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)

    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss


# 建立 VAE 模型
# VAE model
vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# history = vae.fit(x=X_train_flatten, y=X_train_flatten,
#                   shuffle=True,
#                   epochs=1,
#                   batch_size=batch_size,
#                   validation_data=(X_valid_flatten, X_valid_flatten))

# vae.save_weights('vae_mlp_mnist_weights.h5')
vae.load_weights('vae_mlp_mnist_weights.h5')

# 檢視重建圖
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
decoder = Model(decoder_input, _x_decoded_mean)

n = 20  # figure with 20x20 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Construct grid of latent variable values
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# decode for each square in the grid
for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        # z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        # x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.show()

print('end')
