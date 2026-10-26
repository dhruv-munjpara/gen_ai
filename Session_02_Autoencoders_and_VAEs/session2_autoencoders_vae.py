"""
Session 2: Autoencoders & Variational Autoencoders (VAE)
Comprehensive implementation covering:
Task 1: Simple Autoencoder on Fashion MNIST with Reconstruction plots.
Task 2: Loss comparison: Binary Cross-Entropy vs Mean Squared Error.
Task 3: Denoising Autoencoder (Adding Gaussian/Uniform noise and reconstructing clean images).
Task 4: Variational Autoencoder (VAE) with reparameterization trick and latent space sampling.
Task 5: KL-Divergence documentation and metrics.
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

def load_and_preprocess_data():
    print("[INFO] Loading Fashion MNIST dataset...")
    (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    # Normalize to [0, 1] and reshape to (28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # Use subset for fast CPU execution
    x_train_sub = x_train[:10000]
    x_test_sub = x_test[:1000]
    print(f"[INFO] Train data shape: {x_train_sub.shape}, Test data shape: {x_test_sub.shape}")
    return x_train_sub, x_test_sub

# ==========================================
# TASK 1: SIMPLE AUTOENCODER
# ==========================================
def run_task1_simple_autoencoder(x_train, x_test):
    print("\n" + "="*50)
    print("TASK 1: Training Simple Autoencoder (5 epochs)...")
    print("="*50)
    
    input_img = keras.Input(shape=(28, 28, 1))
    # Encoder
    x = layers.Flatten()(input_img)
    x = layers.Dense(128, activation='relu')(x)
    encoded = layers.Dense(64, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Dense(28 * 28, activation='sigmoid')(x)
    decoded = layers.Reshape((28, 28, 1))(x)
    
    autoencoder = keras.Model(input_img, decoded, name="Simple_Autoencoder")
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, shuffle=True, verbose=1)
    
    reconstructed = autoencoder.predict(x_test[:10])
    
    # Plot original vs reconstructed
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    for i in range(10):
        # Original
        axes[0, i].imshow(x_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f"Orig {i+1}", fontsize=8)
        axes[0, i].axis('off')
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"Recon {i+1}", fontsize=8)
        axes[1, i].axis('off')
    
    plt.suptitle("Task 1: Original (Top) vs Autoencoder Reconstructions (Bottom)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task1_reconstructions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Task 1 plot to {out_path}")
    return autoencoder

# ==========================================
# TASK 2: BCE VS MSE LOSS COMPARISON
# ==========================================
def run_task2_loss_comparison(x_train, x_test):
    print("\n" + "="*50)
    print("TASK 2: Training Autoencoder with BCE vs MSE Loss...")
    print("="*50)
    
    def build_model():
        inp = keras.Input(shape=(28, 28, 1))
        x = layers.Flatten()(inp)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(28 * 28, activation='sigmoid')(x)
        out = layers.Reshape((28, 28, 1))(x)
        return keras.Model(inp, out)
    
    # BCE Model
    model_bce = build_model()
    model_bce.compile(optimizer='adam', loss='binary_crossentropy')
    model_bce.fit(x_train, x_train, epochs=3, batch_size=128, verbose=0)
    
    # MSE Model
    model_mse = build_model()
    model_mse.compile(optimizer='adam', loss='mean_squared_error')
    model_mse.fit(x_train, x_train, epochs=3, batch_size=128, verbose=0)
    
    test_samples = x_test[:3]
    pred_bce = model_bce.predict(test_samples)
    pred_mse = model_mse.predict(test_samples)
    
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(3):
        # Original
        axes[i, 0].imshow(test_samples[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f"Original #{i+1}")
        axes[i, 0].axis('off')
        
        # BCE
        axes[i, 1].imshow(pred_bce[i].reshape(28, 28), cmap='gray')
        axes[i, 1].set_title(f"BCE Loss #{i+1}")
        axes[i, 1].axis('off')
        
        # MSE
        axes[i, 2].imshow(pred_mse[i].reshape(28, 28), cmap='gray')
        axes[i, 2].set_title(f"MSE Loss #{i+1}")
        axes[i, 2].axis('off')
        
    plt.suptitle("Task 2: Loss Function Visual Comparison (Original vs BCE vs MSE)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task2_bce_vs_mse_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Task 2 loss comparison plot to {out_path}")

# ==========================================
# TASK 3: DENOISING AUTOENCODER
# ==========================================
def run_task3_denoising_autoencoder(x_train, x_test):
    print("\n" + "="*50)
    print("TASK 3: Training Denoising Autoencoder...")
    print("="*50)
    
    noise_factor = 0.35
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
    
    # Convolutional Denoising Autoencoder
    input_img = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    denoiser = keras.Model(input_img, decoded)
    denoiser.compile(optimizer='adam', loss='binary_crossentropy')
    denoiser.fit(x_train_noisy, x_train, epochs=4, batch_size=128, shuffle=True, verbose=1)
    
    denoised_imgs = denoiser.predict(x_test_noisy[:10])
    
    # Plot 10 images: Original, Noisy, Denoised
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        # Original
        axes[0, i].imshow(x_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f"Orig {i+1}", fontsize=8)
        axes[0, i].axis('off')
        # Noisy
        axes[1, i].imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"Noisy {i+1}", fontsize=8)
        axes[1, i].axis('off')
        # Denoised
        axes[2, i].imshow(denoised_imgs[i].reshape(28, 28), cmap='gray')
        axes[2, i].set_title(f"Denoised {i+1}", fontsize=8)
        axes[2, i].axis('off')
        
    plt.suptitle("Task 3: Denoising Autoencoder (Clean -> Noisy Input -> Denoised Reconstruction)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task3_denoising_results.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Task 3 denoising plot to {out_path}")

# ==========================================
# TASK 4: VARIATIONAL AUTOENCODER (VAE)
# ==========================================
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z (reparameterization trick)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def run_task4_variational_autoencoder(x_train):
    print("\n" + "="*50)
    print("TASK 4: Building & Sampling from Variational Autoencoder (VAE)...")
    print("="*50)
    
    latent_dim = 2
    # Encoder
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(latent_inputs)
    x = layers.Dense(28 * 28, activation="sigmoid")(x)
    decoder_outputs = layers.Reshape((28, 28, 1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    vae.fit(x_train, epochs=4, batch_size=128, verbose=1)
    
    # Sample 5 random points from Standard Normal Prior N(0, I)
    np.random.seed(42)
    random_latents = np.random.normal(loc=0.0, scale=1.0, size=(5, latent_dim))
    generated_images = decoder.predict(random_latents)
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    for i in range(5):
        axes[i].imshow(generated_images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"z=[{random_latents[i,0]:.2f}, {random_latents[i,1]:.2f}]", fontsize=9)
        axes[i].axis('off')
        
    plt.suptitle("Task 4: 5 Novel Images Generated by Sampling VAE Latent Space N(0, I)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task4_vae_generated_samples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Task 4 VAE generated samples plot to {out_path}")

if __name__ == "__main__":
    x_train, x_test = load_and_preprocess_data()
    run_task1_simple_autoencoder(x_train, x_test)
    run_task2_loss_comparison(x_train, x_test)
    run_task3_denoising_autoencoder(x_train, x_test)
    run_task4_variational_autoencoder(x_train)
    print("\n[COMPLETE] All Session 2 tasks executed and visual artifacts generated!")
