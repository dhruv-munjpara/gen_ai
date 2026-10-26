# Session 1: Introduction to Generative Models - Assignment

---

## Task 1: Three Everyday Applications & Proposed Generative Enhancements

### 1. WhatsApp (Smart Conversational Summaries & Catch-Up)
* **Current Experience:** When a user is added to an active group or opens the app after hours, they must scroll through hundreds of unread messages to find important announcements, dates, or decisions.
* **Proposed Generative Feature:** **"Catch-Up Summary" (Local on-device Small Language Model)**.
* **User Experience Improvement:** A clean 3-bullet-point summary pill atop the chat:
  - *Meeting decided for 5:00 PM at Cafe Coffee Day.*
  - *Rohan will bring the presentation slides.*
  - *Split bill settled by Priya.*
  Saves 10-15 minutes of scrolling and prevents missed critical deadlines.

### 2. Google Calendar (Dynamic Natural Language Schedule Synthesizer)
* **Current Experience:** Users must manually click slots, pick dates, set recurring rules, and coordinate invitees' available timezones.
* **Proposed Generative Feature:** **"Conversational Itinerary Generator"**.
* **User Experience Improvement:** The user says: *"Plan a 3-day business trip to Bangalore next Tuesday with 2 client dinners, 4 hours of focused coding each morning, and airport travel buffers."* The generative model synthesizes the optimal block schedule, auto-adjusts for transit, and sends invites with custom agendas.

### 3. Spotify (Personalized Generative DJ & Morning Podcast Producer)
* **Current Experience:** Algorithmic playlists (Discover Weekly) rely on collaborative filtering, giving static tracklists with no conversational narrative.
* **Proposed Generative Feature:** **"AI Hyper-Personalized Morning Host"**.
* **User Experience Improvement:** An AI DJ generates realistic voice intros between songs, weaving together the morning weather forecast, local city news, cricket match highlights, and the musical backstory of the next queued indie track.

---

## Task 2: Comparative Table - Discriminative vs Generative Models

| Aspect / Dimension | Discriminative Models | Generative Models |
| :--- | :--- | :--- |
| **1. Primary Objective** | Model the decision boundary between classes to predict a label $Y$ given an observation $X$. | Model the underlying probability distribution of the data to synthesize new valid instances of $X$. |
| **2. Mathematical Formulation** | Learns conditional probability: $$P(Y \mid X)$$ | Learns joint or marginal probability: $$P(X) \quad \text{or} \quad P(X, Y) = P(X \mid Y)P(Y)$$ |
| **3. Nature of Output** | Discrete class label, categorical probability distribution, or continuous numerical regression target. | High-dimensional data instances: full natural language text, photographic images, audio waveforms, molecular structures. |
| **4. Typical Loss Functions** | Categorical Cross-Entropy, Binary Cross-Entropy (BCE), Hinge Loss, Mean Squared Error (MSE). | Evidence Lower Bound (ELBO), Adversarial Minimax Loss ($V(D, G)$), Variational Diffusion Loss ($L_{\text{diff}}$). |
| **5. Model Architectures** | Logistic Regression, Support Vector Machines (SVM), Random Forests, ResNet, VGG, BERT (for classification). | Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models, GPT / LLaMA / Claude. |
| **6. Common Failure Modes** | Overfitting, vulnerability to adversarial examples/noise, class imbalance bias. | Mode collapse (GANs), hallucinations (LLMs), posterior collapse (VAEs), high compute latency. |

---

## Task 3: ChatGPT Prompt & Generated Marketing Email for Flipkart Tech Gadget

### The Prompt Given to the LLM:
```text
Act as a world-class senior copywriter for Flipkart. Write an engaging, high-converting 
promotional launch email for the newly released "Flipkart SuperPods Pro ANC Earbuds".
Highlight key features: 
- Active Noise Cancellation (ANC up to 45dB)
- 60-hour marathon battery life
- Low-latency gaming mode (35ms)
- Exclusive introductory launch price of ₹2,499 (50% off MRP for Big Billion Days)
Tone: Energetic, tech-savvy, persuasive, and crisp with clear call-to-actions.
```

### The Generated Output:

**Subject Line:** ⚡ Silence the Noise, Unleash the Beat: SuperPods Pro ANC Just Dropped on Flipkart!

**Preview Text:** 45dB Active Noise Cancellation + 60h Battery at an unbeatable launch price of ₹2,499.

```text
Hey Tech Enthusiast,

Tired of ambient traffic buzzing through your favorite playlist? 
Ready to step into pure, uninterrupted sonic bliss?

Meet the all-new Flipkart SuperPods Pro ANC — engineered to transform the way you listen, game, and conquer your day.

Here is why your ears will thank you:
🎧 Silence on Demand: 45dB Hybrid Active Noise Cancellation wipes out background chaos with a single tap.
⚡ 60-Hour Marathon Battery: 10 minutes of FastCharge gives you 5 hours of non-stop playback.
🎮 Ultra-Low 35ms Latency: Hear every footstep before your rivals do with dedicated Gaming Mode.
🌧️ IPX5 Sweat & Splash Resistance: From sweaty gym sessions to monsoon commutes, you are covered.

🔥 EXCLUSIVE LAUNCH OFFER:
MRP: ₹4,999
Launch Day Steal: Just ₹2,499 (Flat 50% OFF)
+ Extra 10% instant discount on Axis & ICICI Bank Cards!

Stock is strictly limited for the launch drop. Don't wait until the countdown ends!

👉 [GRAB YOUR SUPERPODS PRO NOW ON FLIPKART]

Turn down the noise. Turn up your life.

Cheers,
The Flipkart Audio Team
```

---

## Task 4: Detailed Feature Breakdown - Architecture, Flow & Mitigations

### Feature: "AI Product Review Synthesizer & Buyer Q&A on E-Commerce"
1. **Input:** 200+ raw, unstructured customer reviews, ratings, and technical product specifications.
2. **System Architecture:**
   - **Vector Database (Chroma/Pinecone):** Chunk and index reviews with hybrid search (BM25 + Dense embeddings).
   - **Retriever Module:** Filter relevant pros, cons, and durability feedback based on verified purchase tags.
   - **LLM Synthesis Engine:** Summarizes recurring themes into *"What buyers love"*, *"Things to keep in mind"*, and answer user questions in 1-2 concise sentences.
3. **Output:** A real-time structured summary pill on the product page.
4. **Potential Pitfalls & Mitigation Strategies:**
   - **Hallucination Risk:** The LLM might invent specifications (e.g. claiming water resistance when reviews say it stopped working in rain).  
     *Mitigation:* Ground generation strictly via RAG; instruct the prompt to output *"Information not available in verified reviews"* if citations are absent.
   - **Latency Overhead:** Direct LLM calls per page view would crash page load speeds.  
     *Mitigation:* Pre-compute summaries asynchronously via background worker queues upon every 25 new reviews, caching results in Redis.
