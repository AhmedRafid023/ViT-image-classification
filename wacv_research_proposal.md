# Research Proposal: Leveraging Pretrained VLM Projectors for Efficient Vision Transformer Fine-Tuning

**Target Venue:** WACV (Winter Conference on Applications of Computer Vision)  
**Research Area:** Image Classification, Transfer Learning, Efficient Fine-Tuning  

---

## 1. Abstract

We propose a lightweight and effective approach for improving image classification performance using Vision Transformers (ViTs) by incorporating pretrained projectors from Vision-Language Models (VLMs). Our key finding is that inserting a frozen ViT backbone with a trainable VLM projector — originally designed for cross-modal alignment — outperforms full fine-tuning of the ViT backbone while requiring significantly fewer trainable parameters. This work investigates why pretrained VLM projectors transfer effectively to discriminative tasks, how to generalize this approach across architectures with mismatched dimensions, and how it compares to existing parameter-efficient fine-tuning (PEFT) methods such as LoRA and Adapter layers.

---

## 2. Motivation

Large-scale pretrained ViTs have become the backbone of choice for image classification tasks. However, full fine-tuning of these models is computationally expensive, and lightweight alternatives (e.g., linear probing) often leave significant performance on the table.

Concurrently, Vision-Language Models (VLMs) such as LLaVA and Gemma-based multimodal models use **projector modules** to bridge visual encoders and language models. These projectors are trained on massive image-text datasets and are hypothesized to encode rich, transferable visual feature transformations.

We ask: *Can pretrained VLM projectors — originally designed for cross-modal grounding — serve as powerful feature re-mappers that improve discriminative image classification without retraining the visual encoder?*

---

## 3. Preliminary Findings

Experiments were conducted on the **ImageNet** dataset using `google/vit-large-patch32-384` as the visual encoder (VE) and pretrained weights from the **LLaVA projector** (input dimension: 1024, which matches ViT-Large output).

| Configuration | Accuracy |
|---|---|
| VE (frozen) + Classifier Head (CH) — baseline | 86.7% |
| VE (frozen) + Projector (frozen) + CH | 87.5% |
| VE (frozen) + Projector (trainable) + CH | **89.9%** |
| VE (trainable) + CH | 89.6% |
| VE (trainable) + Projector (trainable) + CH | 88.0% |

**Key observations:**

- Training only the projector (`VE frozen + Proj trainable + CH`) achieves the **highest accuracy (89.9%)**, surpassing even full ViT fine-tuning (89.6%).
- This configuration requires **significantly fewer trainable parameters** than fine-tuning the ViT backbone.
- Freezing both the projector and VE still yields a +0.8% gain over the baseline, suggesting pretrained projector weights encode useful visual priors.
- Training all components simultaneously leads to slight degradation, suggesting interference between VE and projector optimization.

---

## 4. Problem Statement

Despite the promising results, the proposed approach has a critical limitation:

> **The dimensional coupling problem:** The current pipeline works because the LLaVA projector's input dimension (1024) coincidentally matches the ViT-Large output dimension (1024). For ViT backbones with different output dimensions (e.g., ViT-Base: 768, Gemma projector input: 1152), a bridge module is required — but naive MLP bridges can degrade performance.

This proposal outlines experiments to:
1. Solve the dimensional coupling problem elegantly.
2. Validate the generalizability of the approach across backbones and projector sources.
3. Explain *why* the pretrained projector helps.
4. Position the method competitively against SOTA parameter-efficient fine-tuning baselines.

---

## 5. Proposed Experiments

### Experiment 1: Dimension Adapter Study
**Goal:** Decouple projector reuse from dimensional constraints.

Test multiple bridging strategies for architecturally mismatched ViT-Projector pairs (e.g., ViT-Base 768 → LLaVA projector 1024):

- Linear projection (no nonlinearity)
- Lightweight 2-layer MLP
- Cross-attention adapter
- Depthwise convolutional bridge

**Expected outcome:** Identify a bridging strategy that preserves projector expressivity without degrading accuracy, converting our limitation into a generalizable design principle.

---

### Experiment 2: Projector Source Ablation *(Critical)*
**Goal:** Determine whether the gain comes from the *architecture* or the *pretrained weights* of the projector.

| Configuration | Purpose |
|---|---|
| LLaVA projector (pretrained weights) | Current best — baseline for comparison |
| LLaVA projector architecture (random init) | Isolates effect of pretraining |
| Gemma projector (pretrained, 1152-dim, with bridge) | Cross-VLM generalization |
| Random MLP of identical size | Architecture vs. weights |
| No projector | Original baseline |

**Hypothesis:** Pretrained weights >> random init of same architecture, supporting the claim that VLM projectors encode *transferable cross-modal visual semantics*.

---

### Experiment 3: Backbone Generalization
**Goal:** Validate that the approach is not specific to one ViT variant.

Test `VE (frozen) + Projector (trainable) + CH` across:

- `google/vit-base-patch16-224` (768-dim)
- `google/vit-large-patch16-224` (1024-dim)
- `facebook/dino-vitb16` (768-dim)
- `openai/clip-vit-large-patch14` (1024-dim)

---

### Experiment 4: Projector Architecture Variants
**Goal:** Determine whether the LLaVA projector design (2-layer MLP + GELU) is optimal or if it can be improved.

Ablations:
- 1-layer vs. 2-layer vs. 3-layer depth
- With / without LayerNorm
- Bottleneck MLP (1024 → 512 → 1024)
- Expansion MLP (1024 → 2048 → 1024)

---

### Experiment 5: Training Efficiency Analysis *(Key Selling Point)*
**Goal:** Quantitatively demonstrate the efficiency advantage.

Metrics to report:
- Number of trainable parameters per configuration
- Wall-clock training time per epoch
- GPU memory consumption
- Accuracy vs. training epochs (convergence curves)

**Expected narrative:** `Proj(train) + CH` achieves superior accuracy to `VE(train) + CH` with ~10–20× fewer trainable parameters.

---

### Experiment 6: Data Efficiency / Few-Shot Study
**Goal:** Test whether pretrained projector weights compensate for limited labeled data.

Train all configurations on ImageNet subsets: 1%, 5%, 10%, 50%, 100%.

**Hypothesis:** `Proj(train) + CH` maintains its advantage under low-data regimes, suggesting the projector encodes useful priors that reduce dependence on labeled examples.

---

### Experiment 7: Comparison Against PEFT Baselines *(Reviewer-Expected)*
**Goal:** Position the method relative to established parameter-efficient fine-tuning methods.

Baselines:
- LoRA (applied to ViT attention layers)
- Adapter layers (Houlsby et al.)
- BitFit (bias-only fine-tuning)
- Visual Prompt Tuning (VPT)
- Linear probing (our baseline)

---

### Experiment 8: Feature Space Analysis *(Qualitative Support)*
**Goal:** Provide interpretable evidence for *why* the projector helps.

- **t-SNE / UMAP visualizations** of feature space before and after the projector — does it improve class separability?
- **Centered Kernel Alignment (CKA)** to measure how much the projector transforms feature geometry.
- **Cross-dataset transfer:** Take the projector trained on ImageNet and evaluate directly on CIFAR-100, Oxford Pets, Food-101 — does it generalize zero-shot?

---

## 6. Proposed Method Overview

```
Input Image
    │
    ▼
┌─────────────────────────┐
│  Visual Encoder (ViT)   │  ← Frozen
│  google/vit-large-p32   │
└─────────────────────────┘
    │  [B, N, 1024]
    ▼
┌──────────────────────────────┐
│  Dimension Bridge (if needed)│  ← Only when dims mismatch
│  Linear / Cross-Attn Adapter │
└──────────────────────────────┘
    │  [B, N, D_proj]
    ▼
┌──────────────────────────────┐
│  VLM Projector               │  ← Trainable (key contribution)
│  LLaVA / Gemma MLP Projector │
└──────────────────────────────┘
    │  [B, N, D_out]
    ▼
┌──────────────────────────────┐
│  Classifier Head             │  ← Trainable
│  Linear / MLP                │
└──────────────────────────────┘
    │
    ▼
Class Prediction
```

---

## 7. Anticipated Contributions

1. **Empirical finding:** Pretrained VLM projectors, when used as trainable adapters on frozen ViTs, outperform full ViT fine-tuning on ImageNet with fewer trainable parameters.
2. **Architectural contribution:** A generalizable dimension adapter design that decouples the method from specific ViT-projector dimension matches.
3. **Analytical contribution:** Ablation evidence distinguishing the role of projector *architecture* vs. pretrained *weights*, providing insight into what VLM projectors learn.
4. **Efficiency contribution:** Quantitative demonstration of training speed, memory, and parameter efficiency advantages over full fine-tuning and PEFT baselines.

---

## 8. Addressing Anticipated Reviewer Concerns

| Reviewer Concern | Mitigation |
|---|---|
| "Dimension match is coincidental" | Experiment 1 — dimension adapter study |
| "Only one projector tested" | Experiment 2 — projector source ablation |
| "Only one ViT backbone tested" | Experiment 3 — backbone generalization |
| "No comparison to PEFT methods" | Experiment 7 — LoRA, Adapter, BitFit comparison |
| "Why does the projector help?" | Experiment 8 — t-SNE, CKA, cross-dataset transfer |
| "Limited practical impact" | Experiment 5 & 6 — efficiency and few-shot results |

---

## 9. Proposed Narrative for WACV

> *"We show that projector modules from pretrained Vision-Language Models encode transferable visual semantics that, when fine-tuned as lightweight adapters on frozen ViTs, achieve state-of-the-art classification accuracy while requiring an order of magnitude fewer trainable parameters than full backbone fine-tuning. We further introduce a generalizable dimension adapter design that decouples our method from architectural constraints, enabling broad applicability across ViT variants and VLM projector sources."*

---

## 10. Experiment Priority (for Time-Constrained Schedule)

| Priority | Experiment | Rationale |
|---|---|---|
| 1 | Projector source ablation | Most critical theoretical claim |
| 2 | Training efficiency analysis | Easiest win, strong selling point |
| 3 | PEFT baseline comparison | Expected by reviewers |
| 4 | Dimension adapter study | Directly fixes stated limitation |
| 5 | t-SNE / feature analysis | Strong qualitative support |
| 6 | Backbone generalization | Demonstrates breadth |
| 7 | Few-shot / data efficiency | Bonus contribution |

---

*Proposal drafted for WACV submission. All experiments are designed to be run on ImageNet unless otherwise specified.*
