# Session 5: Applying Generative Models in Industry - Assignment

---

## Tasks & Artifact Overview

| Task | Title | Description | Artifact |
| :--- | :--- | :--- | :--- |
| **Task 1** | IPL Match Article & LLM Summary | 3-paragraph dramatic cricket final news report + 1-paragraph LLM summary. | [`ipl_summary.txt`](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/ipl_summary.txt) |
| **Task 2** | Zomato Restaurant Q&A Chatbot | Python dictionary database with budget, cuisine, rating & delivery querying. | [`session5_industrial_apps.py`](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/session5_industrial_apps.py#L22-L108) |
| **Task 3** | Futuristic Gujarati Thali Generation | Prompt: *"A futuristic Gujarati thali with neon colors"*. | [`futuristic_gujarati_thali.png`](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/futuristic_gujarati_thali.png) |
| **Task 4** | Spotify Song Filter ('S' letter) | Copilot/ChatGPT prompt, Python function, and unit test validation. | Documented below in Section 4. |
| **Task 5** | Production Chatbot Challenges | Latency & concurrency under flash sales + Hallucination prevention. | Documented below in Section 5. |

---

## Task 3: Image Synthesis & Cultural-Futuristic Analysis

### The Prompt:
```text
"A futuristic Gujarati thali with neon colors"
```

### Generated Artifact:
- Filename: [`futuristic_gujarati_thali.png`](file:///d:/DA&DS/gen_ai/Session_05_Applying_Generative_Models_in_Industry/futuristic_gujarati_thali.png)

### In-Depth Analysis:
1. **Tradition vs Cyberpunk Fusion:** The visual synthesizes centuries-old culinary heritage (an authentic round *kansa* or stainless steel platter surrounded by traditional *katoris*) with cybernetic visual aesthetics (glowing cyan edge-lit boundaries, holographic grids, and luminescent food textures).
2. **Harmonious Culinary Layout:** The layout faithfully preserves the traditional spatial geometry of a Gujarati meal:
   - **Center Anchor:** The staple millet *Bajra Rotla* / *Thepla* topped with a luminous white dollop of freshly churned butter (*Makhan*).
   - **Radial Symmetry:** 8 encircling bowls containing signature items: spicy-sweet *Undhiyu*, sour-sweet Gujarati *Dal*, buttermilk *Kadhi*, tangy *Sev Tameta*, yellow savory *Dhokla*, rolled *Khandvi*, saffron-infused *Shrikhand*, and chilled *Chhas*.
3. **Lighting & Contrast:** High saturation neon green, magenta, amber, and electric cyan illuminate each dish against a deep navy matte background, turning a heritage dining feast into a futuristic sci-fi culinary journey.

---

## Task 4: GitHub Copilot / ChatGPT Prompt & Spotify Playlist Filter

### Prompt Submitted to AI Assistant:
```text
Write a clean, efficient, and robust Python function called `filter_songs_starting_with_s` 
that takes a list of song titles (from a Spotify playlist) and returns a new list containing 
only those songs whose names start with the letter 'S' or 's'.
Requirements:
1. Case-insensitive matching ('S' and 's').
2. Handle and strip leading whitespaces gracefully.
3. Validate that input is a list of strings.
4. Include type hints and clean docstrings.
```

### Generated Python Function:
```python
def filter_songs_starting_with_s(playlist: list[str]) -> list[str]:
    """
    Filters a list of song titles (case-insensitive) and returns only
    those starting with the letter 'S', stripping leading whitespace.
    """
    if not isinstance(playlist, list):
        raise TypeError("Input playlist must be a list of song title strings.")
    
    return [
        song for song in playlist 
        if isinstance(song, str) and song.strip().upper().startswith("S")
    ]
```

### Test Cases & Output:
```python
playlist = [
    "Shape of You", "Blinding Lights", "Starboy", "stay", 
    "Apna Bana Le", "Something Just Like This", "Kesariya", 
    "Sunflower", "Viva La Vida", "  shivers"
]

# Output:
['Shape of You', 'Starboy', 'stay', 'Something Just Like This', 'Sunflower', '  shivers']
# All 6 songs starting with 'S' / 's' captured correctly.
```

---

## Task 5: Production Challenges for GenAI Chatbots (BookMyShow / Myntra) & Solutions

### Challenge 1: Severe Latency & Concurrency Bottlenecks During Flash Spikes
* **Scenario:** On **BookMyShow** during Coldplay / World Cup ticket drops, or on **Myntra** during the Big Fashion Festival, traffic spikes from 5,000 to over 500,000 active concurrent users per minute.
* **Problem:** Large Language Models (LLMs) have high inference latency (typically 500ms to 2.5 seconds per query). If every customer queries the GenAI assistant simultaneously, GPU inference queues saturate, causing timeouts, dropped sessions, and server crashes.
* **Engineering Solutions:**
  1. **Tiered Architecture (Hybrid Intent Router):** Deploy a lightweight, sub-10ms intent classifier (e.g., DistilBERT or ONNX-quantized Small Language Model) at the edge. Route 80% of repetitive, deterministic queries (e.g., *"Where is my ticket QR?"*, *"What time does gate open?"*) to instant cached static templates.
  2. **Semantic Caching with Vector DBs:** Use Redis / Milvus semantic caching (`GPTCache`). If a user asks a semantically identical question answered in the last 60 seconds (cosine similarity $> 0.95$), return the cached response without invoking GPU inference.
  3. **Speculative Decoding & Model Quantization (vLLM / TensorRT-LLM):** Serve models using AWQ 4-bit quantization and continuous batching engines, boosting token throughput by 4x to 6x on the same hardware footprint.

---

### Challenge 2: Hallucination of Critical Commercial Information (Inventory & Refund Policies)
* **Scenario:** A user asks the Myntra support bot: *"Can I return this customized designer blazer after 45 days for a 100% cash refund?"* An ungrounded generative model might hallucinate: *"Yes, we offer 60 days hassle-free cash refunds on all items!"*, legally exposing the company.
* **Problem:** Foundation LLMs generate plausible-sounding text based on statistical token probabilities rather than authoritative company policy databases.
* **Engineering Solutions:**
  1. **Strict Retrieval-Augmented Generation (RAG) with Fact-Checking Guardrails:**
     - Restrict the LLM's system prompt to answer *strictly* using cited snippets from authoritative knowledge bases (e.g. Myntra Return Policy v3.2).
     - Add explicit instruction: *"If the retrieved policy context does not explicitly mention the item category, state: 'Please refer to the item's product page for specific return exclusions.'"*
  2. **Automated Post-Generation Guardrails (NeMo Guardrails / Guardrails AI):**
     - Run a secondary lightweight verification pass before streaming text to the user.
     - Validate numeric claims (prices, return windows, seat availability) against live SQL/REST APIs (e.g., verifying BookMyShow seat booking status via microservice APIs before promising a seat).
  3. **Deterministic Tool Use (Strict Function Calling):** Force the LLM to output structured JSON tool calls (e.g. `check_refund_eligibility(order_id)`) rather than inventing transaction policies in free-form prose.
