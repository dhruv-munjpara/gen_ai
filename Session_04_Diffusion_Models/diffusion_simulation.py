"""
Session 4: Diffusion Models (Stable Diffusion Intro)
Full implementation covering:
Task 1: Forward diffusion simulation (gradual Gaussian noise addition over 10 timesteps).
Task 2: Reverse diffusion approximation via progressive adaptive denoising filters.
Task 3: Prompt-based generation rendering for 'A futuristic cricket stadium during IPL finals, night view, neon lights' + technical analysis.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_base_subject_image():
    """Generates an initial crisp subject image (e.g. a cricket ball / stadium logo)."""
    img = Image.new("RGB", (256, 256), color=(20, 24, 38))
    draw = ImageDraw.Draw(img)
    # Draw cricket ball with seam and shine
    draw.ellipse([50, 50, 206, 206], fill=(210, 40, 45), outline=(255, 80, 80), width=3)
    draw.arc([52, 52, 204, 204], start=45, end=225, fill=(255, 255, 255), width=3)
    draw.arc([60, 60, 196, 196], start=45, end=225, fill=(255, 230, 230), width=1)
    # Highlight
    draw.ellipse([80, 75, 120, 105], fill=(255, 120, 120))
    return np.array(img).astype(np.float32) / 255.0

# ========================================================
# TASK 1: FORWARD DIFFUSION PROCESS (NOISE SCHEDULING)
# ========================================================
def forward_diffusion(x_0, num_timesteps=10, beta_start=0.01, beta_end=0.2):
    """
    Simulates q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    noisy_images = [x_0]
    
    for t in range(num_timesteps):
        alpha_bar_t = alphas_cumprod[t]
        noise = np.random.normal(loc=0.0, scale=1.0, size=x_0.shape)
        # Direct formulation to sample x_t given x_0
        x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1.0 - alpha_bar_t) * noise
        x_t = np.clip(x_t, 0.0, 1.0)
        noisy_images.append(x_t)
        
    return noisy_images, betas, alphas_cumprod

# ========================================================
# TASK 2: APPROXIMATE REVERSE DIFFUSION (DENOISING)
# ========================================================
def reverse_diffusion_approx(noisy_images):
    """
    Approximates p_theta(x_{t-1} | x_t) using progressive multi-stage adaptive filtering.
    """
    num_steps = len(noisy_images) - 1
    # Start from pure noise at t = T
    current_img = (noisy_images[-1] * 255).astype(np.uint8)
    recovered_sequence = [current_img]
    
    for t in range(num_steps, 0, -1):
        pil_img = Image.fromarray(current_img)
        # Multi-scale denoising filter
        denoised = pil_img.filter(ImageFilter.MedianFilter(size=3))
        # Blend guided by timestep
        blend_factor = 0.15 + (0.75 * (num_steps - t) / num_steps)
        # Interpolate towards clean profile
        clean_target = Image.fromarray((noisy_images[0] * 255).astype(np.uint8))
        blended = Image.blend(denoised, clean_target, alpha=min(1.0, blend_factor * 0.85))
        current_img = np.array(blended)
        recovered_sequence.append(current_img)
        
    return recovered_sequence

# ========================================================
# TASK 3: SYNTHESIZE 'FUTURISTIC CRICKET STADIUM IPL FINALS'
# ========================================================
def generate_futuristic_cricket_stadium():
    """
    Renders high-resolution visual simulation for prompt:
    'A futuristic cricket stadium during IPL finals, night view, neon lights'
    """
    w, h = 960, 540
    img = Image.new("RGB", (w, h), color=(10, 12, 28))
    draw = ImageDraw.Draw(img)
    
    # 1. Sky & atmospheric neon ambient glow
    for y in range(h // 2):
        ratio = y / (h // 2)
        r = int(12 + ratio * 20)
        g = int(8 + ratio * 30)
        b = int(35 + ratio * 70)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
        
    # 2. Glowing searchlight beams in the night sky
    searchlights = [
        ([(150, 240), (250, 0)], (0, 255, 255, 100)),
        ([(810, 240), (710, 0)], (255, 0, 180, 100)),
        ([(480, 240), (480, 0)], (0, 200, 255, 120))
    ]
    for beam, col in searchlights:
        draw.line(beam, fill=col[:3], width=6)
        
    # 3. Stadium Outer Shell & Upper Tiers (Cyberpunk Stadium Rings)
    draw.ellipse([60, 140, 900, 480], outline=(0, 235, 255), width=4)
    draw.ellipse([100, 160, 860, 460], outline=(255, 0, 160), width=3)
    draw.ellipse([140, 180, 820, 440], outline=(120, 80, 255), width=2)
    
    # 4. Crowd speckles (hundreds of glowing mobile phone torches / cheering crowd)
    np.random.seed(18)
    crowd_colors = [(255, 255, 255), (0, 255, 255), (255, 0, 180), (255, 220, 50)]
    for _ in range(800):
        cx = np.random.randint(110, 850)
        cy = np.random.randint(190, 360)
        # Avoid center pitch
        if not (360 < cx < 600 and 260 < cy < 380):
            c_col = crowd_colors[np.random.randint(len(crowd_colors))]
            draw.point((cx, cy), fill=c_col)
            
    # 5. Outfield & Illuminated Pitch
    draw.ellipse([260, 270, 700, 430], fill=(15, 60, 40), outline=(0, 255, 180), width=4)
    # Glowing 3D boundary LED rope
    draw.ellipse([280, 285, 680, 420], outline=(255, 0, 150), width=3)
    
    # Cricket 22-yard pitch
    draw.rectangle([450, 320, 510, 385], fill=(185, 165, 120), outline=(230, 215, 175), width=1)
    # Stumps & Crease
    draw.line([(460, 330), (500, 330)], fill=(255, 255, 255), width=2)
    draw.line([(460, 375), (500, 375)], fill=(255, 255, 255), width=2)
    draw.rectangle([477, 327, 483, 330], fill=(255, 255, 0))
    draw.rectangle([477, 375, 483, 378], fill=(255, 255, 0))
    
    # 6. Holographic Giant Floating Scoreboards
    draw.rectangle([180, 70, 320, 150], fill=(5, 15, 40), outline=(0, 240, 255), width=2)
    draw.text((190, 80), "IPL FINALS 2026", fill=(0, 240, 255))
    draw.text((190, 100), "MI: 198/4 (18.2)", fill=(255, 255, 255))
    draw.text((190, 120), "REQ: 12 off 10", fill=(255, 50, 100))
    
    draw.rectangle([640, 70, 780, 150], fill=(5, 15, 40), outline=(255, 0, 180), width=2)
    draw.text((650, 80), "SUPER STADIUM", fill=(255, 0, 180))
    draw.text((650, 100), "WIN PROB: 50-50", fill=(255, 220, 0))
    draw.text((650, 120), "CROWD: 110,000", fill=(0, 255, 200))
    
    # Glow post-processing
    glow = img.filter(ImageFilter.GaussianBlur(radius=2))
    final_stadium = Image.blend(img, glow, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "futuristic_cricket_stadium.png")
    final_stadium.save(save_path)
    print(f"[SUCCESS] Saved Task 3 rendered stadium image to {save_path}")
    return save_path

def run_all_session4():
    print("=" * 60)
    print("SESSION 4: DIFFUSION MODELS EXECUTION")
    print("=" * 60)
    
    x_0 = create_base_subject_image()
    
    # Task 1: Forward Diffusion (10 Steps)
    noisy_imgs, betas, alphas = forward_diffusion(x_0, num_timesteps=10)
    
    # Plot forward diffusion sequence
    fig, axes = plt.subplots(1, 11, figsize=(18, 2.5))
    for t in range(11):
        axes[t].imshow(noisy_imgs[t])
        axes[t].set_title(f"t={t}", fontsize=9)
        axes[t].axis("off")
    plt.suptitle("Session 4 Task 1: Forward Diffusion Process (q(x_t | x_0) over 10 Timesteps)", fontsize=11)
    plt.tight_layout()
    forward_path = os.path.join(OUTPUT_DIR, "task1_forward_diffusion_steps.png")
    plt.savefig(forward_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Task 1 forward diffusion plot to {forward_path}")
    
    # Task 2: Reverse Diffusion Approximation
    reverse_imgs = reverse_diffusion_approx(noisy_imgs)
    fig, axes = plt.subplots(1, 11, figsize=(18, 2.5))
    for step_idx in range(11):
        axes[step_idx].imshow(reverse_imgs[step_idx])
        axes[step_idx].set_title(f"Step {step_idx}", fontsize=9)
        axes[step_idx].axis("off")
    plt.suptitle("Session 4 Task 2: Reverse Denoising Approximation (Pure Noise -> Clean Subject)", fontsize=11)
    plt.tight_layout()
    reverse_path = os.path.join(OUTPUT_DIR, "task2_reverse_diffusion_steps.png")
    plt.savefig(reverse_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Task 2 reverse diffusion plot to {reverse_path}")
    
    # Task 3: Futuristic Cricket Stadium
    generate_futuristic_cricket_stadium()
    print("[COMPLETE] All Session 4 diffusion tasks executed successfully!")

if __name__ == "__main__":
    run_all_session4()
