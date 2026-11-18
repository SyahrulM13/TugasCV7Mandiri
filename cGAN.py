
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Pengaturan
batch_size = 200
epochs = 20000
z_dim = 100

# Noise dan label untuk visualisasi
z_vis = tf.random.normal([10, z_dim])
y_vis = tf.constant(np.eye(10), dtype=tf.float32)

# Load dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0

y_train = tf.one_hot(
    y_train,
    depth=10,
    dtype=tf.float32
)

# Dataset untuk training
data_iter = iter(
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(4 * batch_size)
    .batch(batch_size)
    .repeat()
)


# =========================
# GENERATOR
# =========================
def Generator():

    z = tf.keras.layers.Input(shape=(z_dim,), dtype=tf.float32)
    y = tf.keras.layers.Input(shape=(10,), dtype=tf.float32)

    # Gabungkan noise dan label
    x = tf.keras.layers.Concatenate()([z, y])

    x = tf.keras.layers.Dense(3 * 3 * 512)(x)
    x = tf.keras.layers.Reshape((3, 3, 512))(x)

    x = tf.keras.layers.Conv2DTranspose(
        256, 3, strides=2, padding="valid"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(
        128, 4, strides=2, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding="same", activation="sigmoid"
    )(x)

    out = tf.keras.layers.Reshape((28, 28))(x)

    return tf.keras.Model(
        inputs=[z, y],
        outputs=out,
        name="Generator"
    )


# =========================
# DISCRIMINATOR
# =========================
def Discriminator():

    X = tf.keras.layers.Input(shape=(28, 28), dtype=tf.float32)
    Y = tf.keras.layers.Input(shape=(10,), dtype=tf.float32)

    # Ubah label menjadi ukuran 28 x 28
    y = tf.keras.layers.Reshape((1, 1, 10))(Y)
    y = tf.keras.layers.UpSampling2D(size=(28, 28))(y)

    # Ubah gambar menjadi format channel
    x = tf.keras.layers.Reshape((28, 28, 1))(X)

    # Gabungkan gambar dengan label
    x = tf.keras.layers.Concatenate()([x, y])

    x = tf.keras.layers.Conv2D(
        128, 4, strides=2, padding="same"
    )(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    x = tf.keras.layers.Conv2D(
        256, 4, strides=2, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    x = tf.keras.layers.Conv2D(
        512, 4, strides=2, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    # Ratakan fitur sebelum klasifikasi
    x = tf.keras.layers.Flatten()(x)

    # Output berupa nilai real/fake
    out = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(
        inputs=[X, Y],
        outputs=out,
        name="Discriminator"
    )


# Buat model
G = Generator()
D = Discriminator()

# Tampilkan ringkasan model
G.summary()
D.summary()


# =========================
# LOSS FUNCTIONS
# =========================
cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def G_loss(D, x_fake, y):
    fake_output = D([x_fake, y], training=True)

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


def D_loss(D, x_real, x_fake, y):
    real_output = D([x_real, y], training=True)
    fake_output = D([x_fake, y], training=True)

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


# =========================
# OPTIMIZERS
# =========================
G_opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
D_opt = tf.keras.optimizers.Adam(learning_rate=2e-4)


# =========================
# TRAINING
# =========================
for epoch in range(epochs):

    # Ambil noise acak
    z_mb = tf.random.normal([batch_size, z_dim])

    # Ambil data asli dan label
    x_real, y = next(data_iter)

    # Hitung loss dan gradien
    with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:

        # Buat gambar palsu
        x_fake = G([z_mb, y], training=True)

        # Hitung loss Generator dan Discriminator
        G_loss_curr = G_loss(D, x_fake, y)
        D_loss_curr = D_loss(D, x_real, x_fake, y)

    # Hitung gradien
    G_grad = G_tape.gradient(
        G_loss_curr,
        G.trainable_variables
    )

    D_grad = D_tape.gradient(
        D_loss_curr,
        D.trainable_variables
    )

    # Update bobot model
    G_opt.apply_gradients(
        zip(G_grad, G.trainable_variables)
    )

    D_opt.apply_gradients(
        zip(D_grad, D.trainable_variables)
    )

    # Tampilkan hasil setiap 100 epoch
    if epoch % 100 == 0:

        print(
            "epoch: {}; G_loss: {:.6f}; D_loss: {:.6f}".format(
                epoch + 1,
                float(G_loss_curr),
                float(D_loss_curr)
            )
        )

        # Generate gambar angka 0 sampai 9
        generated_images = G(
            [z_vis, y_vis],
            training=False
        )

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))

        for i, ax in enumerate(axes.flat):
            ax.imshow(
                generated_images[i].numpy(),
                cmap="gray",
                vmin=0,
                vmax=1
            )
            ax.set_title("Angka {}".format(i))
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)
