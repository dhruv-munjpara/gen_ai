"""
Session 3: Generative Adversarial Networks (GANs)
Full implementation covering:
Task 1: simple_generator(latent_dim) -> (28, 28) image array.
Task 2: Discriminator class with discriminate(image) returning probability [0, 1].
Task 3: Full adversarial training cycle simulation with loss computation and visual plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================================================
# TASK 1: SIMPLE GENERATOR
# ========================================================
def build_generator_model(latent_dim=100):
    model = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, 128)),
        # Upsample to 14x14
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        # Upsample to 28x28
        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")
    ], name="Generator")
    return model

def simple_generator(noise_vector, generator_model=None):
    """
    Takes a random noise vector of shape (latent_dim,) or (batch, latent_dim)
    and returns a 28x28 synthetic image array.
    """
    if generator_model is None:
        generator_model = build_generator_model(latent_dim=len(noise_vector))
    
    if len(noise_vector.shape) == 1:
        noise_input = np.expand_dims(noise_vector, axis=0)
    else:
        noise_input = noise_vector
        
    generated_tensor = generator_model(noise_input, training=False)
    image_28x28 = generated_tensor.numpy()[0, :, :, 0]
    return image_28x28

# ========================================================
# TASK 2: DISCRIMINATOR CLASS
# ========================================================
class Discriminator:
    """
    Discriminator class evaluating whether a 28x28 image is Real (1.0) or Fake (0.0).
    """
    def __init__(self):
        self.model = keras.Sequential([
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ], name="Discriminator")
        
        self.optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def discriminate(self, image):
        """
        Takes a 28x28 image (numpy array or tensor) and returns probability score in [0, 1].
        """
        if len(image.shape) == 2:
            img_tensor = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            img_tensor = np.expand_dims(image, axis=0)
        else:
            img_tensor = image
            
        prob = self.model(img_tensor, training=False).numpy().flatten()[0]
        return float(prob)

# ========================================================
# TASK 3: SIMULATE ONE COMPLETE ADVERSARIAL CYCLE
# ========================================================
def simulate_adversarial_cycle():
    print("=" * 60)
    print("TASK 3: Simulating One Cycle of Adversarial Training")
    print("=" * 60)
    
    latent_dim = 100
    batch_size = 16
    
    generator = build_generator_model(latent_dim)
    g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    
    discriminator_obj = Discriminator()
    discriminator = discriminator_obj.model
    d_optimizer = discriminator_obj.optimizer
    loss_fn = discriminator_obj.loss_fn
    
    # 1. Prepare simulated real images (e.g. from Fashion MNIST or synthetic distribution [-1, 1])
    real_images = np.random.uniform(-1.0, 1.0, size=(batch_size, 28, 28, 1)).astype("float32")
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))
    
    # 2. Sample random noise for Generator
    noise = tf.random.normal(shape=(batch_size, latent_dim))
    
    # Step A: Train Discriminator
    with tf.GradientTape() as d_tape:
        generated_fake_images = generator(noise, training=True)
        real_preds = discriminator(real_images, training=True)
        fake_preds = discriminator(generated_fake_images, training=True)
        
        d_loss_real = loss_fn(real_labels, real_preds)
        d_loss_fake = loss_fn(fake_labels, fake_preds)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # Step B: Train Generator (Trick Discriminator to classify fakes as Real = 1.0)
    misleading_labels = tf.ones((batch_size, 1))
    with tf.GradientTape() as g_tape:
        generated_fake_images = generator(noise, training=True)
        fake_preds_for_g = discriminator(generated_fake_images, training=True)
        g_loss = loss_fn(misleading_labels, fake_preds_for_g)
        
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    print(f"[CYCLE 1 RESULT] Discriminator Real Loss : {d_loss_real.numpy():.4f}")
    print(f"[CYCLE 1 RESULT] Discriminator Fake Loss : {d_loss_fake.numpy():.4f}")
    print(f"[CYCLE 1 RESULT] Total Discriminator Loss: {d_loss.numpy():.4f}")
    print(f"[CYCLE 1 RESULT] Total Generator Loss    : {g_loss.numpy():.4f}")
    
    # Test single function and class
    test_noise = np.random.normal(0, 1, size=(latent_dim,))
    single_fake_img = simple_generator(test_noise, generator)
    prob_score = discriminator_obj.discriminate(single_fake_img)
    print(f"[DISCRIMINATOR INFERENCE] Evaluated Single Synthetic Image -> Real Probability: {prob_score:.4f}")
    
    # Plot generated synthetic samples
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = generated_fake_images[i].numpy().reshape(28, 28)
        # Rescale from [-1, 1] to [0, 1] for display
        img = (img + 1.0) / 2.0
        p = discriminator_obj.discriminate(img)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Fake #{i+1} (p_real={p:.2f})", fontsize=9)
        ax.axis("off")
        
    plt.suptitle("Session 3 Task 3: Synthetic 28x28 Images Generated by Generator in Cycle 1", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "gan_adversarial_cycle_samples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved GAN adversarial cycle plot to {out_path}")
    return d_loss.numpy(), g_loss.numpy()

if __name__ == "__main__":
    simulate_adversarial_cycle()
