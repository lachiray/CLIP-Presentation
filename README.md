# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever — *OpenAI (2021)*  
**ArXiv:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

---

## 1. Overview and Motivation

Traditional computer vision models (ResNet, EfficientNet) depend on *fixed label sets* like ImageNet, restricting generalization to unseen visual concepts. CLIP introduces a new paradigm: **learning directly from natural language supervision**.

### The Problem
- Conventional models: require expensive human-annotated labels.
- Internet data: contains billions of (image, text) pairs offering free supervision.

### The CLIP Solution
Train dual encoders — one for **images** and one for **text** — to align representations in a shared embedding space using a **contrastive loss**.  
This enables **zero-shot classification**, e.g. distinguishing between:
> “A photo of a cat” vs. “A photo of a dog” without any task-specific fine-tuning.

Dataset: 400M image–text pairs collected from the web.

---

## 2. Architecture Overview and Pseudocode

CLIP jointly trains a vision encoder (ResNet or ViT) and a text encoder (Transformer) via contrastive learning.

### Algorithm
```python
# CLIP Core Algorithm
# Batch of (image, text) pairs
I_f = image_encoder(images)      # [N, d_i]
T_f = text_encoder(texts)        # [N, d_t]
I_e = normalize(I_f @ W_i)       # Map to embedding space
T_e = normalize(T_f @ W_t)
logits = (I_e @ T_e.T) * exp(t)  # Similarity matrix
labels = arange(N)
loss_i = CE(logits, labels)
loss_t = CE(logits.T, labels)
loss = (loss_i + loss_t) / 2
```

**Key Differences from Prior Work:**
- Uses **contrastive objectives** instead of classification or caption prediction.
- Constructs a **joint embedding space** across vision and language.
- Leverages **prompt engineering** (e.g., “a photo of a {label}”) to improve zero-shot generalization.

---

## Code Demonstration

See `clip_demo.ipynb` for a full demonstration using a pre-trained CLIP model.

### What the Demo Shows

#### Loading Pre-trained CLIP Model
- Loads OpenAI’s CLIP ViT-B/32 model.  
- Architecture: Vision Transformer (image encoder) + Transformer (text encoder).  
- Joint embedding dimension: 512.  
- Enables zero-shot evaluation on any image–text pair.

#### Image–Text Similarity Computation
- Two input images (`kiki` and `kiki2`) show the same cat (“Kiki”) in different states — curled up and yawning.  
- The model encodes each image and a set of descriptive text prompts.  
- Both image and text embeddings are projected into a shared latent space, and cosine similarity is computed.

<p align="center">
  <img src="figures/kiki.png" width="600" alt="Input images used in the CLIP demo">
</p>

#### Zero-Shot Classification Example
- Each image is evaluated against natural-language prompts such as:
["a photo of a cat yawning",
"a photo of a sleeping cat",
"a photo of a cat sitting",
"a photo of a dog",
"a photo of a tiger"]

- CLIP outputs probability distributions for how well each text description matches the image.  
- This demonstrates CLIP’s ability to recognize both **objects** and **actions** (yawning vs. sleeping) without any fine-tuning.

<p align="center">
<img src="figures/output.png" width="600" alt="CLIP demo output showing probability distribution">
</p>

#### Key Takeaways
- CLIP aligns visual and linguistic representations through contrastive learning.  
- It generalizes beyond fixed labels, interpreting images using flexible natural-language descriptions.  
- The demo illustrates how CLIP captures semantic nuance — understanding that both photos depict a *cat*, yet identifying distinct behaviors purely from text context.

---

## 4. Critical Analysis

| Aspect | Observation |
|:--|:--|
| **Strength** | Scales efficiently to 400M examples, enabling transfer across 30+ datasets. |
| **Weakness** | Dataset bias and ethical risks from uncurated internet data. |
| **Overlooked Area** | Struggles on specialized or fine-grained datasets (e.g., EuroSAT, medical). |
| **Follow-up Work** | OpenCLIP (LAION), ALIGN (Google), LiT (Google Brain) improved scale and filtering. |

**Discussion Question:** Does web-scale data represent broad human knowledge or only online bias?

---

## 5. Impacts

- **Short-term:** CLIP made zero-shot classification practical for the first time.
- **Long-term:** Foundation for vision–language systems (DALL·E, Flamingo, GPT-4V).
- **Cross-disciplinary:** Enables retrieval, captioning, and visual-language reasoning.

> “CLIP transformed text into a universal labeling interface for vision models.”

---

## 6. Questions for Discussion

**Q1 – Conceptual:**  Why does CLIP’s contrastive loss outperform caption-prediction objectives for transfer learning?

**Q2 – Analytical:**  How does CLIP handle ambiguous labels (e.g., “crane” the bird vs. “crane” the machine)?  
*Hint: prompt engineering and context-driven descriptions.*

---

## 7. Resource Links

| Resource | Link |
|:--|:--|
| Paper (ArXiv) | [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) |
| OpenAI CLIP GitHub | [https://github.com/openai/CLIP](https://github.com/openai/CLIP) |
| OpenCLIP (LAION) | [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| CLIP Blog | [https://openai.com/research/clip](https://openai.com/research/clip) |

---

## 8. Citation

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning Transferable Visual Models from Natural Language Supervision.* arXiv preprint arXiv:2103.00020.
