'''
Implementation of Variational-Autoencoder model in Python using Keras and TensorFlow as backend. I have used Fashion-MNIST dataset, which can be easily grabbed from
https://github.com/zalandoresearch/fashion-mnist . The "Encoder" part uses Convolutional neural network to reduce each original 28X28 image into a data-point in a two-dimensional space.
Then, the "Decoder" part of code samples points from this two-dimensional space based on a Normal Distribution and tries to re-construct original images from them.
'''

from utils.mnist_reader import load_mnist
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, BatchNormalization, Reshape, LeakyReLU, Activation, UpSampling2D
import keras.backend as K
from keras.objectives import binary_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Importing train and test data from cloned Fashion-MNIST git repository
X_main, y_main = load_mnist("data/fashion", kind="train")
X_main = X_main.astype(np.float32)
y_main = y_main.astype(np.float32)

X_test, y_test = load_mnist("data/fashion", kind="t10k")
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)


# Reshaping all images to 28X28 format for convolutional layer.
X_main = X_main.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))


# Adding horizontally flipped version of each image to the train set.
X_flipped = np.zeros(X_main.shape)
for img in range(60000):
    X_flipped[img, :, :, 0] = np.fliplr(X_main[img, :, :, 0])
    
X_train = np.concatenate([X_main, X_flipped], axis=0)


X_train /= 255.
X_test /= 255.
n_epochs = 50
batch_size = 100


# Encoder part of network.
inp = Input((28, 28, 1))
conv1 = Conv2D(32, 4, padding='same')(inp)
batch1 = BatchNormalization()(conv1)
lr1 = Activation('relu')(batch1)
conv2 = Conv2D(64, 4, strides=2, padding='same')(lr1)
batch2 = BatchNormalization()(conv2)
lr2 = Activation('relu')(batch2)
conv3 = Conv2D(128, 4, padding='same')(lr2)
batch3 = BatchNormalization()(conv3)
lr3 = Activation('relu')(batch3)
conv4 = Conv2D(256, 4, strides=2, padding='same')(lr3)
batch4 = BatchNormalization()(conv4)
lr4 = Activation('relu')(batch4)
flat = Flatten()(lr4)
h1 = Dense(256)(flat)
batch5 = BatchNormalization()(h1)
lr5 = Activation('relu')(batch5)
z_mean = Dense(2)(lr5)
z_log_sigma = Dense(2)(lr5)


# Sampling layer for sampling of points from two-dimensional latent space.
def sampling(args):
    
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2), mean=0.0, stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mean, z_log_sigma])


# Layers for the Decoder part of network.
h3 = Dense(49, activation='relu')
h3_reshape = Reshape((7, 7, 1))
conv_trans1 = Conv2D(256, 4, padding='same')
batch6 = BatchNormalization()
lr6 = Activation('relu')
conv_trans2 = Conv2D(128, 4, padding='same')
batch7 = BatchNormalization()
lr7 = Activation('relu')
up1 = UpSampling2D()
conv_trans3 = Conv2D(64, 4, padding='same')
batch8 = BatchNormalization()
lr8 = Activation('relu')
conv_trans4 = Conv2D(32, 4, padding='same')
batch9 = BatchNormalization()
lr9 = Activation('relu')
up2 = UpSampling2D()
out = Conv2D(1, 3, padding='same')
batch10 = BatchNormalization()
lr10 = Activation('sigmoid')


# Decoder model for training of whole network i.e. from Decoder input (original image) to Encoder output (generated output).
h3_vae = h3(z)
h3_vae_reshape = h3_reshape(h3_vae)
conv_trans1_vae = conv_trans1(h3_vae_reshape)
batch6_vae = batch6(conv_trans1_vae)
lr6_vae = lr6(batch6_vae)
conv_trans2_vae = conv_trans2(lr6_vae)
batch7_vae = batch7(conv_trans2_vae)
lr7_vae = lr7(batch7_vae)
up1_vae = up1(lr7_vae)
conv_trans3_vae = conv_trans3(up1_vae)
batch8_vae = batch8(conv_trans3_vae)
lr8_vae = lr8(batch8_vae)
conv_trans4_vae = conv_trans4(lr8_vae)
batch9_vae = batch9(conv_trans4_vae)
lr9_vae = lr9(batch9_vae)
up2_vae = up2(lr9_vae)
out_vae = out(up2_vae)
batch10_vae = batch10(out_vae)
lr10_vae = lr10(batch10_vae)

vae_comp = Model(inp, lr10_vae)					# Complete model
inp_to_latent = Model(inp, z_mean)				# Model for representing original images in two-dimensional latent space

# Decoder model from latent input to generated output of Decoder
gen_inp = Input((2,))
h3_gen = h3(gen_inp)
h3_gen_reshape = h3_reshape(h3_gen)
conv_trans1_gen = conv_trans1(h3_gen_reshape)
batch6_gen = batch6(conv_trans1_gen)
lr6_gen = lr6(batch6_gen)
conv_trans2_gen = conv_trans2(lr6_gen)
batch7_gen = batch7(conv_trans2_gen)
lr7_gen = lr7(batch7_gen)
up1_gen = up1(lr7_gen)
conv_trans3_gen = conv_trans3(up1_gen)
batch8_gen = batch8(conv_trans3_gen)
lr8_gen = lr8(batch8_gen)
conv_trans4_gen = conv_trans4(lr8_gen)
batch9_gen = batch9(conv_trans4_gen)
lr9_gen = lr9(batch9_gen)
up2_gen = up2(lr9_gen)
out_gen = out(up2_gen)
batch10_gen = batch10(out_gen)
lr10_gen = lr10(batch10_gen)

gen_model = Model(gen_inp, lr10_gen)			# Generator model


# Loss for training whole model, sum of cross-entropy between input and output and kl-divergence.
def vae_loss(inp, out):
    
    print (inp.shape, out.shape)
    xent_loss = 28 * 28 * binary_crossentropy(K.flatten(inp), K.flatten(out))
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

# Compiling the complete model (I started with Adam optimizer for faster convergence and later on switched to SGD with nesterov with lower learning rate)
#vae_comp.compile(optimizer=Adam(lr=0.005), loss=vae_loss)
vae_comp.compile(optimizer=SGD(lr=0.0001, momentum=0.5, nesterov=True), loss=vae_loss)

# Running the training model
vae_comp.fit(X_train, X_train, batch_size=batch_size, epochs=n_epochs, validation_data=(X_test, X_test), callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1)])


# Prediction of values in latent space for the test-set of 10,000 images.
encod_model = Model(inp, z_mean)
x_test_latent = inp_to_latent.predict(X_test, batch_size=batch_size)
latent_fig = plt.figure(figsize=(10,10))
plt.scatter(x_test_latent[:, 0], x_test_latent[:, 1], c=y_test)
plt.colorbar()
latent_fig.savefig('latent_repr.png', bbox_inches='tight')
plt.show()


scaler = MinMaxScaler(feature_range=(0, 255))


# Generating images using Generator model from Gaussian noise having zero mean and unit variance.
g_input = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, 2))
preds = gen_model.predict(g_input)
for img in range(100):
    image = preds[img, :, :, 0]
    preds[img, :, :, 0] = scaler.fit_transform(image)


fig = plt.figure(figsize=(10, 10))
for img in range(100):
    ax = fig.add_subplot(10, 10, img+1)
    ax = plt.imshow(preds[img, :, :, 0])
	
fig.savefig('gen_sample.png', bbox_inches='tight')
plt.show()