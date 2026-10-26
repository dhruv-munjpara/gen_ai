# Introduction to Generative AI - Coursework & Assignments

Repository containing complete, session-wise solutions, mathematical derivations, runnable implementations, and generated visual artifacts for the **Introduction to Gen AI** module.

---

## 📁 Repository Structure & Directory Navigation

| Module / Session Folder | Description | Key Artifacts & Executables |
| :--- | :--- | :--- |
| [What_is_GenAI](file:///d:/DA&DS/gen_ai/What_is_GenAI/) | Foundational concepts of Generative AI, 5-aspect comparative table, analysis of 5 real-world tools, and food app poem analysis. | [README.md](file:///d:/DA&DS/gen_ai/What_is_GenAI/README.md), [demo_genai_poem.py](file:///d:/DA&DS/gen_ai/What_is_GenAI/demo_genai_poem.py) |
| [Comparison_with_Traditional_AI_ML](file:///d:/DA&DS/gen_ai/Comparison_with_Traditional_AI_ML/) | In-depth comparison across data, compute, metrics, interpretability & business fit; Zomato case study; synthetic feature transformation; Flipkart support bot dialogue. | [README.md](file:///d:/DA&DS/gen_ai/Comparison_with_Traditional_AI_ML/README.md), [traditional_vs_genai_sim.py](file:///d:/DA&DS/gen_ai/Comparison_with_Traditional_AI_ML/traditional_vs_genai_sim.py) |
| [Session_01_Introduction_to_Generative_Models](file:///d:/DA&DS/gen_ai/Session_01_Introduction_to_Generative_Models/) | 3 everyday app generative enhancements (WhatsApp, Calendar, Spotify DJ), Discriminative vs Generative table, Flipkart gadget launch marketing prompt & email, feature breakdown. | [README.md](file:///d:/DA&DS/gen_ai/Session_01_Introduction_to_Generative_Models/README.md) |
| [Session_02_Autoencoders_and_VAEs](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/) | Trained Autoencoders and VAE on Fashion MNIST: BCE vs MSE loss comparison, Denoising Autoencoder ($\sigma=0.35$), VAE latent space sampling $\mathcal{N}(0, I)$, and KL-Divergence derivation. | [session2_autoencoders_vae.py](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/session2_autoencoders_vae.py), [task1_reconstructions.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task1_reconstructions.png), [task2_bce_vs_mse_comparison.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task2_bce_vs_mse_comparison.png), [task3_denoising_results.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task3_denoising_results.png), [task4_vae_generated_samples.png](file:///d:/DA&DS/gen_ai/Session_02_Autoencoders_and_VAEs/task4_vae_generated_samples.png) |
| [Session_03_Generative_Adversarial_Networks_GANs](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/) | Simple generator function, Discriminator class, adversarial minimax loop simulation with loss logging, real-world apps (Pix2Pix, StyleGAN try-on), and DCGAN vs StyleGAN vs CycleGAN analysis. | [gan_implementation.py](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/gan_implementation.py), [gan_adversarial_cycle_samples.png](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/gan_adversarial_cycle_samples.png), [README.md](file:///d:/DA&DS/gen_ai/Session_03_Generative_Adversarial_Networks_GANs/README.md) |
| [Session_04_Diffusion_Models](file:///d:/DA&DS/gen_ai/Session_04_Diffusion_Models/) | Forward diffusion noise scheduling ($q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ over 10 steps), reverse denoising approximation, synthesized *"Futuristic IPL cricket stadium at night with neon lights"*, industrial inpainting/super-resolution, and C2PA ethical safeguards. | [diffusion_simulation.py](file:///d:/DA&DS/gen_ai/Session_04_Diffusion_Models/diffusion_simulation.py), [task1_forward_diffusion_steps.png](file:///d:/DA&DS/gen_ai/Session_04_Diffusion_Models/task1_forward_diffusion_steps.png), [task2_reverse_diffusion_steps.png](file:///d:/DA&DS/gen_ai/Session_04_Diffusion_Models/task2_reverse_diffusion_steps.png), [futuristic_cricket_stadium.png](file:///d:/DA&DS/gen_ai/Session_04_Diffusion_Models/futuristic_cricket_stadium.png), [README.md](file:///d:/DA&DS/gen_ai/Session_04_Diffusion_Models/README.md) |
| [Session_05_Applying_Generative_Models_in_Industry](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/) | Dramatic IPL match news article & LLM summary, Zomato-style Q&A assistant with budget/cuisine/ratings matching, synthesized *"Futuristic Gujarati Thali"* rendering, Copilot Spotify 'S'-song filter, and enterprise production solutions for BookMyShow & Myntra. | [ipl_summary.txt](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/ipl_summary.txt), [session5_industrial_apps.py](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/session5_industrial_apps.py), [futuristic_gujarati_thali.png](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/futuristic_gujarati_thali.png), [README.md](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/README.md) |

---

## 🚀 Running Any Session Script

All scripts are standalone and executable directly from PowerShell or Command Prompt:

```bash
# Session 2: Autoencoders and VAEs
cd d:\DA&DS\gen_ai\Session_02_Autoencoders_and_VAEs
python session2_autoencoders_vae.py

# Session 3: GAN Generator, Discriminator & Adversarial Loop
cd d:\DA&DS\gen_ai\Session_03_Generative_Adversarial_Networks_GANs
python gan_implementation.py

# Session 4: Forward & Reverse Diffusion + Stadium Synthesis
cd d:\DA&DS\gen_ai\Session_04_Diffusion_Models
python diffusion_simulation.py

# Session 5: Zomato Chatbot, Thali Synthesis & Spotify Song Filter
cd d:\DA&DS\gen_ai\Session_05_Applying_Generative_Models_in_Industry
python session5_industrial_apps.py
```
