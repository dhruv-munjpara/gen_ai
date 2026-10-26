# Session 2: Autoencoders & Variational Autoencoders (VAE) - Assignment

---

## Overview of Tasks & Artifacts

| Task | Title | Description | Artifact |
| :--- | :--- | :--- | :--- |
| **Task 1** | Simple Autoencoder | Keras Autoencoder trained on Fashion MNIST (5 epochs), displaying original vs reconstructed. | [task1_reconstructions.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task1_reconstructions.png) |
| **Task 2** | BCE vs MSE Loss Comparison | Visual reconstruction differences using Binary Cross-Entropy vs Mean Squared Error. | [task2_bce_vs_mse_comparison.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task2_bce_vs_mse_comparison.png) |
| **Task 3** | Denoising Autoencoder | Gaussian noise addition ($\sigma=0.35$) and convolutional clean image reconstruction. | [task3_denoising_results.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task3_denoising_results.png) |
| **Task 4** | Variational Autoencoder (VAE) | Reparameterization trick, latent Gaussian prior $\mathcal{N}(0, I)$, and 5 sampled synthetic items. | [task4_vae_generated_samples.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task4_vae_generated_samples.png) |
| **Task 5** | KL-Divergence Analysis | Mathematical formulation, role in preventing cluster gaps, and regularization impact. | Documented below in Section 5. |

---

## Task 1: Simple Autoencoder Architecture & Training

### Architecture
- **Input:** $28 \times 28 \times 1$ grayscale Fashion MNIST images normalized to $[0, 1]$.
- **Encoder:** Flatten $\to$ Dense(128, ReLU) $\to$ Dense(64, ReLU) bottleneck representation.
- **Decoder:** Dense(128, ReLU) $\to$ Dense(784, Sigmoid) $\to$ Reshape to $(28, 28, 1)$.
- **Optimizer & Loss:** Adam ($lr=0.001$), Binary Cross-Entropy.
- **Training Duration:** 5 Epochs, Batch Size 128.

### Results
The model successfully compresses the 784-dimensional pixel space into a 64-dimensional latent bottleneck and recovers outlines of t-shirts, trousers, pullovers, and shoes.

---

## Task 2: Binary Cross-Entropy (BCE) vs Mean Squared Error (MSE) Loss

### 1. Conceptual Comparison
* **Binary Cross-Entropy (BCE):**
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i) \right]$$
  - Treats each normalized pixel intensity as a Bernoulli probability of being "active" (white/gray).
  - Penalizes confident incorrect predictions exponentially, resulting in crisper edges and higher contrast transitions between object boundaries and dark backgrounds.
* **Mean Squared Error (MSE):**
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$
  - Minimizing $L_2$ distance corresponds to estimating the conditional Gaussian mean of pixel intensities.
  - Tends to produce slightly blurrier reconstructions because averaging multiple plausible pixel values minimizes squared error at the cost of sharpness.

---

## Task 3: Denoising Autoencoder

### Methodology
1. **Noise Injection:** Added Gaussian noise ($\mathcal{N}(0, 0.35^2)$) to clean test images and clipped pixel values to $[0.0, 1.0]$.
2. **Convolutional Denoiser Architecture:**
   - 2D Conv layers ($3\times3$, 32 filters) + MaxPooling $(2\times2)$ downsampling.
   - 2D Conv layers ($3\times3$, 32 filters) + UpSampling $(2\times2)$ upsampling.
   - Final Conv2D layer with Sigmoid activation.
3. **Outcome:** Even when severe grain obscures textures (e.g. shoe straps, shirt folds), the network restores topological continuity and suppresses high-frequency noise artifacts.

---

## Task 4: Variational Autoencoder (VAE) Latent Space Sampling

### Why Standard Autoencoders Fail at Generation
A standard autoencoder maps inputs to deterministic, isolated points in latent space. The empty regions between trained clusters produce distorted or meaningless outputs when sampled.

### VAE Mechanism
- Instead of encoding an input as a single deterministic vector $z$, the VAE encoder outputs parameters of a probability distribution: **mean vector $\mu$** and **log-variance vector $\log(\sigma^2)$**.
- **Reparameterization Trick:**
  $$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
  This allows backpropagation through stochastic nodes.
- **Generation:** 5 random points were sampled directly from the standard normal prior $\mathcal{N}(0, I)$ and fed through the decoder, producing novel Fashion MNIST items.

---

## Task 5: Deep Dive into KL-Divergence Loss in VAE

### The Prompt Given / Technical Question:
> *"Explain what Kullback-Leibler (KL) Divergence loss is in the context of a Variational Autoencoder (VAE), mathematically derive its role in the loss function, and explain why a VAE cannot function as a generative model without it."*

### Detailed Technical Explanation:

#### 1. What is KL-Divergence?
Kullback-Leibler (KL) Divergence is an asymmetric statistical measure of how one probability distribution $q(x)$ diverges from a reference probability distribution $p(x)$:
$$D_{\text{KL}}(q(z \mid x) \parallel p(z)) = \int q(z \mid x) \log \left( \frac{q(z \mid x)}{p(z)} \right) dz$$

In a VAE:
- $q_\phi(z \mid x)$ is the **approximate posterior** produced by the encoder.
- $p(z) = \mathcal{N}(0, I)$ is the **prior distribution** (standard isotropic Gaussian).

#### 2. The Complete VAE Loss Function (ELBO)
The objective minimized during training is the negative Evidence Lower Bound:
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{Reconstruction}}(x, \hat{x}) + D_{\text{KL}}\left( q_\phi(z \mid x) \parallel \mathcal{N}(0, I) \right)$$

For Gaussian distributions with diagonal covariance, the KL-divergence has a closed-form analytical solution:
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

#### 3. Why is KL-Divergence Essential for Generative Capabilities?
Without the KL-divergence loss term:
1. **Latent Space Fragmentation:** The encoder would minimize reconstruction loss by placing clusters arbitrarily far apart in latent space with infinitesimally small variance ($\sigma \to 0$). The space would be riddled with vast "voids".
2. **Inability to Sample:** When you want to generate a new image at test time, you sample from $\mathcal{N}(0, I)$. If the encoder did not force the latent distributions to conform to $\mathcal{N}(0, I)$, sampling a random vector from $\mathcal{N}(0, I)$ would land in an empty "void", yielding nonsensical noise.
3. **Smooth Interpolation:** KL-divergence acts as a continuous regularizer that forces latent distributions to overlap smoothly, ensuring that walking between two points in latent space yields a smooth, continuous semantic transition (e.g. smoothly morphing a t-shirt into a jacket).
