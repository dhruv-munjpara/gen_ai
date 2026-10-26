# Session 3: Generative Adversarial Networks (GANs) - Assignment

---

## Tasks & Artifact Overview

| Task | Title | Description | Code / Artifact |
| :--- | :--- | :--- | :--- |
| **Task 1** | Simple Generator Function | `simple_generator(noise_vector)` mapping latent noise to a $(28, 28)$ image array. | [gan_implementation.py](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/gan_implementation.py#L38-L53) |
| **Task 2** | Discriminator Class | `Discriminator` class with `.discriminate(image)` returning probability $\in [0, 1]$. | [gan_implementation.py](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/gan_implementation.py#L58-L87) |
| **Task 3** | Adversarial Cycle Simulation | One complete forward/backward cycle for both Generator and Discriminator with loss logging. | [gan_adversarial_cycle_samples.png](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/gan_adversarial_cycle_samples.png) |
| **Task 4** | Real-World GAN Applications | Image-to-Image translation (Pix2Pix/CycleGAN) and Style Transfer / Virtual Try-On. | Documented below in Section 4. |
| **Task 5** | DCGAN vs StyleGAN vs CycleGAN | Architecture evolution, style disentanglement, and cycle-consistency loss. | Documented below in Section 5. |

---

## Task 1 & 2: Architecture Details

### Generator Architecture
- **Input:** 100-dimensional continuous noise vector $\mathbf{z} \sim \mathcal{N}(0, I)$.
- **Dense Projection & Reshape:** Dense layer expanding to $7 \times 7 \times 128 = 6,272$ features.
- **Transposed Convolutions:**
  - Conv2DTranspose (64 filters, kernel $4\times4$, stride 2, LeakyReLU $\alpha=0.2$) $\to 14 \times 14 \times 64$.
  - Conv2DTranspose (1 filter, kernel $4\times4$, stride 2, Tanh activation) $\to 28 \times 28 \times 1$.

### Discriminator Architecture
- **Input:** $28 \times 28 \times 1$ image tensor.
- **Strided Convolutions & Regularization:**
  - Conv2D (64 filters, kernel $3\times3$, stride 2, LeakyReLU $\alpha=0.2$, Dropout 0.3) $\to 14 \times 14 \times 64$.
  - Conv2D (128 filters, kernel $3\times3$, stride 2, LeakyReLU $\alpha=0.2$, Dropout 0.3) $\to 7 \times 7 \times 128$.
  - Flatten $\to$ Dense(1, Sigmoid) outputting $P(\text{Image is Real})$.

---

## Task 3: Adversarial Training Cycle Mathematics & Execution

### The Two-Player Minimax Game
$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

1. **Discriminator Update:**
   - Real Loss: $-\log D(\mathbf{x}_{\text{real}})$
   - Fake Loss: $-\log(1 - D(G(\mathbf{z})))$
   - Update $D$ to maximize classification accuracy.
2. **Generator Update:**
   - Non-saturating heuristic loss: $-\log D(G(\mathbf{z}))$
   - Update $G$ so that $D$ outputs 1.0 (Real) for synthetic images.

---

## Task 4: Real-World Applications of GANs

### 1. Medical Imaging & Cross-Modality Image-to-Image Translation (Pix2Pix / CycleGAN)
- **Application:** Translating between magnetic resonance imaging (MRI) and computed tomography (CT) scans.
- **Value:** Obtaining a CT scan exposes patients to ionizing radiation, while MRI does not. GANs trained on paired or unpaired scans synthesize realistic synthetic CT (sCT) images directly from MRIs, enabling accurate radiation therapy planning without extra radiation exposure.
- **Commercial Tool:** SyntheticMR, subtlePET.

### 2. Virtual Fashion Try-On & E-Commerce Catalog Generation (StyleGAN / VTON)
- **Application:** Synthesizing photorealistic images of apparel on diverse human models of varying body shapes, skin tones, and poses without expensive studio photoshoot logistics.
- **Value:** Myntra and Amazon use GAN-based pose-guided image generation to drape dresses and shirts naturally over uploaded user selfies, slashing return rates by over 30%.
- **Commercial Tool:** Lalaland.ai, Veesual.

---

## Task 5: Architectural Evolution: DCGAN vs StyleGAN vs CycleGAN

| Feature / Dimension | DCGAN (Deep Convolutional GAN) | StyleGAN (Style-based GAN) | CycleGAN (Cycle-Consistent GAN) |
| :--- | :--- | :--- | :--- |
| **Year & Authors** | 2015 (Radford, Metz, Chintala) | 2018–2021 (Karras et al., NVIDIA) | 2017 (Zhu et al., UC Berkeley) |
| **Core Innovation** | Replaced fully connected layers with all-convolutional nets, batch normalization, and strided convolutions. | Style-based generator architecture; maps noise to intermediate latent space $\mathcal{W}$ using an 8-layer MLP and AdaIN (Adaptive Instance Norm). | Unpaired image-to-image translation using two generators ($G: X \to Y$, $F: Y \to X$) and two discriminators with **Cycle-Consistency Loss**. |
| **Latent Control** | Entangled latent space $\mathcal{Z}$. Changing one variable alters multiple unrelated visual attributes. | Highly disentangled $\mathcal{W}$ space. Enables granular, independent control over coarse (pose, face shape), middle (hair, facial features), and fine styles (color, micro-textures). | Conditioned on source domain images rather than random Gaussian noise; no paired training data required ($X \leftrightarrow Y$). |
| **Key Loss Function** | Standard Binary Cross-Entropy Minimax Loss. | Non-saturating logistic loss with $R_1$ path-length regularization. | Adversarial Loss + Cycle Consistency Loss: $$\mathcal{L}_{\text{cyc}} = \|F(G(x)) - x\|_1 + \|G(F(y)) - y\|_1$$ |
| **Primary Use Case** | Baseline image synthesis (e.g. low-res $64\times64$ bedrooms, digits). | Ultra-high resolution ($1024\times1024$) photorealistic human faces (*ThisPersonDoesNotExist.com*), artistic portraits. | Domain translation: Horse $\leftrightarrow$ Zebra, Monet painting $\leftrightarrow$ Photo, Summer $\leftrightarrow$ Winter landscapes. |

### Summary of Key Learnings:
1. **Adversarial Equilibrium is Fragile:** GAN training suffers from mode collapse (where the generator produces only a few repetitive realistic samples) and gradient vanishing if the discriminator overpowers the generator early on.
2. **Disentanglement is Critical:** StyleGAN demonstrated that routing noise through a mapping network prevents the distribution from being constrained by prior Gaussian geometry, enabling surgical attribute manipulation.
3. **Cycle Consistency Enables Unsupervised Translation:** CycleGAN proved that pairing constraints can be replaced by the mathematical requirement that translating forward and then backward must return to the original input ($x \to y \to x$).
