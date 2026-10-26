"""
Session 5: Applying Generative Models in Industry
Implementations:
Task 2: Zomato-style Restaurant Q&A Chatbot using knowledge dictionary & fuzzy intent parsing.
Task 3: Synthesis of 'A futuristic Gujarati thali with neon colors' image.
Task 4: Spotify playlist song filter (songs starting with 'S') with prompt, tests, and benchmarks.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================================================
# TASK 2: ZOMATO-STYLE RESTAURANT Q&A CHATBOT
# ========================================================
RESTAURANT_DATABASE = {
    "spice_villa": {
        "name": "Spice Villa Grand",
        "cuisine": ["North Indian", "Mughlai", "Biryani"],
        "rating": 4.6,
        "delivery_time_mins": 35,
        "price_for_two": 650,
        "top_dishes": ["Butter Chicken", "Dal Makhani", "Dum Biryani", "Garlic Naan"],
        "opening_hours": "11:00 AM - 11:30 PM",
        "location": "SG Highway, Ahmedabad",
        "offers": "Flat 20% off with code TASTE20"
    },
    "sasumaa_thali": {
        "name": "Sasumaa Gujarati Thali",
        "cuisine": ["Authentic Gujarati", "Kathiyawadi", "Vegetarian"],
        "rating": 4.8,
        "delivery_time_mins": 25,
        "price_for_two": 450,
        "top_dishes": ["Undhiyu", "Sev Tameta", "Rotla with White Butter", "Rasawala Dhokla", "Shrikhand"],
        "opening_hours": "11:30 AM - 3:30 PM, 7:00 PM - 10:30 PM",
        "location": "C.G. Road, Navrangpura",
        "offers": "Unlimited buttermilk on dine-in; 15% off on orders above Rs. 500"
    },
    "wok_express": {
        "name": "Wok & Roll Pan Asian",
        "cuisine": ["Chinese", "Thai", "Asian Fusion"],
        "rating": 4.4,
        "delivery_time_mins": 30,
        "price_for_two": 550,
        "top_dishes": ["Hakka Noodles", "Crispy Chilli Paneer", "Thai Green Curry", "Dim Sums"],
        "opening_hours": "12:00 PM - 11:00 PM",
        "location": "Prahlad Nagar",
        "offers": "Free Fortune Cookie with every bowl"
    },
    "caffe_roma": {
        "name": "Caffe Roma Artisan Pizzeria",
        "cuisine": ["Italian", "Wood-fired Pizza", "Pasta"],
        "rating": 4.7,
        "delivery_time_mins": 40,
        "price_for_two": 800,
        "top_dishes": ["Truffle Mushroom Pizza", "Penne Arrabbiata", "Classic Tiramisu"],
        "opening_hours": "12:00 PM - 12:00 AM",
        "location": "Bodakdev",
        "offers": "Buy 1 Large Pizza Get 1 Garlic Bread Free"
    }
}

class ZomatoQAChatbot:
    """
    Intelligent restaurant assistant providing answers on cuisines, ratings,
    delivery times, budget constraints, and recommendations.
    """
    def __init__(self, db):
        self.db = db

    def query(self, user_question: str) -> str:
        q = user_question.lower().strip()
        
        # 1. Budget filtering
        budget_match = re.search(r"under\s*(?:rs\.?|inr)?\s*(\d+)", q)
        if budget_match:
            max_budget = int(budget_match.group(1))
            matching = [r for r in self.db.values() if r["price_for_two"] <= max_budget]
            if matching:
                lines = [f"- {r['name']} ({', '.join(r['cuisine'])}): Rs. {r['price_for_two']} for two | Rating: {r['rating']} stars" for r in matching]
                return f"Here are the top restaurants under Rs. {max_budget} for two:\n" + "\n".join(lines)
            return f"I couldn't find restaurants under Rs. {max_budget}. Our most affordable is {min(self.db.values(), key=lambda x: x['price_for_two'])['name']} at Rs. 450 for two."

        # 2. Cuisine search
        for cuisine_key in ["gujarati", "kathiyawadi", "north indian", "biryani", "chinese", "asian", "italian", "pizza"]:
            if cuisine_key in q:
                matched = [r for r in self.db.values() if any(cuisine_key in c.lower() for c in r["cuisine"])]
                if matched:
                    r = matched[0]
                    return (
                        f"For delicious {cuisine_key.title()} food, I highly recommend '{r['name']}' in {r['location']}.\n"
                        f"Rating: {r['rating']} stars | Delivery: ~{r['delivery_time_mins']} mins | Price for two: Rs. {r['price_for_two']}\n"
                        f"Must-try dishes: {', '.join(r['top_dishes'][:3])}\n"
                        f"Active offer: {r['offers']}"
                    )

        # 3. Fast delivery
        if "fastest" in q or "quickest" in q or "fast delivery" in q:
            fastest = min(self.db.values(), key=lambda x: x["delivery_time_mins"])
            return f"The fastest delivery is '{fastest['name']}' taking only {fastest['delivery_time_mins']} minutes! They serve {', '.join(fastest['cuisine'])}."

        # 4. Highest rated
        if "best" in q or "top rated" in q or "highest rating" in q or "highest rated" in q:
            best = max(self.db.values(), key=lambda x: x["rating"])
            return (
                f"Our top-rated spot is '{best['name']}' with an exceptional {best['rating']} stars rating!\n"
                f"Famous for: {', '.join(best['top_dishes'])}\n"
                f"Location: {best['location']}"
            )

        # Default fallback
        names = ", ".join([r["name"] for r in self.db.values()])
        return f"I can help you explore restaurants like {names}. Ask me for recommendations by cuisine, price (e.g. 'under 500'), fastest delivery, or top ratings!"

# ========================================================
# TASK 3: SYNTHESIZE 'FUTURISTIC GUJARATI THALI'
# ========================================================
def generate_futuristic_gujarati_thali():
    """
    Renders high-resolution image for prompt:
    'A futuristic Gujarati thali with neon colors'
    """
    w, h = 800, 800
    img = Image.new("RGB", (w, h), color=(12, 14, 24))
    draw = ImageDraw.Draw(img)
    
    # Background holographic grid
    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill=(20, 25, 45), width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill=(20, 25, 45), width=1)
        
    center_x, center_y = w // 2, h // 2
    
    # Outer Large Stainless Steel / Holographic Neon Thali (Tray)
    outer_radius = 350
    draw.ellipse(
        [center_x - outer_radius, center_y - outer_radius, center_x + outer_radius, center_y + outer_radius],
        fill=(25, 28, 48),
        outline=(0, 255, 230),
        width=6
    )
    # Secondary interior neon rim
    draw.ellipse(
        [center_x - outer_radius + 15, center_y - outer_radius + 15, center_x + outer_radius - 15, center_y + outer_radius - 15],
        outline=(255, 0, 160),
        width=3
    )
    
    # Center: Rotla / Thepla / Puri with glowing white butter (Makhan)
    center_radius = 110
    draw.ellipse(
        [center_x - center_radius, center_y - center_radius, center_x + center_radius, center_y + center_radius],
        fill=(80, 65, 40),
        outline=(255, 200, 80),
        width=3
    )
    # Glowing Makhan (butter dollop)
    draw.ellipse(
        [center_x - 30, center_y - 25, center_x + 30, center_y + 25],
        fill=(255, 255, 230),
        outline=(0, 255, 255),
        width=2
    )
    draw.text((center_x - 38, center_y - 8), "NEO-ROTLA", fill=(40, 30, 10))
    
    # Surrounding Katoris (Bowls) with traditional dishes in neon aesthetics
    katoris = [
        ("UNDHIYU", (60, 140, 50), (0, 255, 120), -90),
        ("DAL", (210, 160, 20), (255, 220, 0), -45),
        ("KADHI", (230, 230, 150), (255, 255, 100), 0),
        ("SEV TAMETA", (210, 50, 40), (255, 60, 80), 45),
        ("DHOKLA", (240, 200, 30), (255, 240, 0), 90),
        ("KHANDVI", (220, 180, 40), (255, 180, 0), 135),
        ("SHRIKHAND", (190, 80, 190), (255, 0, 230), 180),
        ("CHHAS", (240, 255, 255), (0, 200, 255), 225),
    ]
    
    ring_radius = 235
    bowl_r = 52
    for name, fill_col, neon_col, angle_deg in katoris:
        rad = np.radians(angle_deg)
        bx = int(center_x + ring_radius * np.cos(rad))
        by = int(center_y + ring_radius * np.sin(rad))
        
        # Draw Bowl
        draw.ellipse([bx - bowl_r, by - bowl_r, bx + bowl_r, by + bowl_r], fill=fill_col, outline=neon_col, width=4)
        # Dish name
        draw.text((bx - 28, by - 6), name, fill=(255, 255, 255))
        
    # Title Header Banner
    draw.rectangle([180, 15, 620, 55], fill=(10, 15, 35), outline=(0, 255, 230), width=2)
    draw.text((220, 25), "FUTURISTIC GUJARATI THALI 2050", fill=(0, 255, 230))
    
    # Soft Neon Blur Glow
    glow = img.filter(ImageFilter.GaussianBlur(radius=3))
    final_thali = Image.blend(img, glow, alpha=0.35)
    
    save_path = os.path.join(OUTPUT_DIR, "futuristic_gujarati_thali.png")
    final_thali.save(save_path)
    print(f"[SUCCESS] Saved Task 3 Gujarati thali image to {save_path}")
    return save_path

# ========================================================
# TASK 4: SPOTIFY PLAYLIST 'S' FILTER FUNCTION
# ========================================================
def filter_songs_starting_with_s(playlist: list[str]) -> list[str]:
    """
    Filters a list of song titles (case-insensitive) and returns only
    those starting with the letter 'S', stripping leading whitespace.
    """
    if not isinstance(playlist, list):
        raise TypeError("Input playlist must be a list of song title strings.")
    
    return [song for song in playlist if isinstance(song, str) and song.strip().upper().startswith("S")]

def test_song_filter():
    print("\n" + "=" * 60)
    print("TASK 4: Testing Spotify Playlist Song Filter")
    print("=" * 60)
    sample_playlist = [
        "Shape of You",
        "Blinding Lights",
        "Starboy",
        "stay",
        "Apna Bana Le",
        "Something Just Like This",
        "Kesariya",
        "Sunflower",
        "Viva La Vida",
        "  shivers"
    ]
    filtered = filter_songs_starting_with_s(sample_playlist)
    print("Original Playlist :", sample_playlist)
    print("Filtered ('S' Only):", filtered)
    assert len(filtered) == 6, f"Expected 6 songs, got {len(filtered)}"
    print("[TEST PASSED] Successfully filtered all songs starting with 'S'!")

if __name__ == "__main__":
    # Test Chatbot
    print("=" * 60)
    print("TASK 2: ZOMATO CHATBOT DEMONSTRATION")
    print("=" * 60)
    bot = ZomatoQAChatbot(RESTAURANT_DATABASE)
    queries = [
        "Can you recommend an authentic Gujarati place?",
        "What are the best options under 500 for two?",
        "Who offers the fastest delivery?",
        "What is your highest rated restaurant?"
    ]
    for q in queries:
        print(f"\nUser: '{q}'")
        print(bot.query(q))
        
    # Task 3
    generate_futuristic_gujarati_thali()
    
    # Task 4
    test_song_filter()
