## Appendix: Architectural Components and Dimensional Specifications

To ensure the theoretical integrity of our proposal and prevent any language-aligned data leakage, we explicitly isolate our experimental choices. The vision backbones are trained strictly via visual supervised or self-supervised paradigms (excluding text-aligned models like CLIP), while the target VLM projectors encompass a diverse mix of MLP and attention-based cross-modal mappers.

---

### 1. Vision Transformer (ViT) Backbones
We evaluate our method across three distinct visual paradigms: **Supervised Learning (Vanilla)**, **Self-Supervised Contrastive/Distillation (DINOv2)**, and **Masked Image Modeling (EVA-02)**.

| Backbone Family | HuggingFace / `timm` Identifier | Pretraining Paradigm | Output Dimension ($D_{\text{in}}$) |
| :--- | :--- | :--- | :--- |
| **Vanilla ViT-Base** | `google/vit-base-patch16-224` | Supervised (ImageNet-21k) | **768** |
| **Vanilla ViT-Large** | `google/vit-large-patch16-224` | Supervised (ImageNet-21k) | **1024** |
| **Vanilla ViT-Huge** | `google/vit-huge-patch14-224-in21k` | Supervised (ImageNet-21k) | **1280** |
| **DINOv2-Base** | `facebook/dinov2-base` | Self-Supervised (Discriminative) | **768** |
| **DINOv2-Large** | `facebook/dinov2-large` | Self-Supervised (Discriminative) | **1024** |
| **EVA-02-Base** | `timm/eva02_base_patch14_224` | Masked Image Modeling (MIM) | **768** |
| **EVA-02-Large** | `timm/eva02_large_patch14_224` | Masked Image Modeling (MIM) | **1024** |

---

### 2. Vision-Language Model (VLM) Projectors
We sample pretrained projection layers from state-of-the-art multimodal families to evaluate cross-VLM generalization. These modules span across simple MLPs, optimized single linear layers, and cross-attention abstractors.

| Projector Source Family | Core Architecture Type | Target Projector Input ($D_{\text{proj}}$) | Target Projector Output ($D_{\text{out}}$) |
| :--- | :--- | :--- | :--- |
| **LLaVA 1.5** | 2-Layer MLP + GELU | **1024** | **4096** (Vicuna / LLaMA-2 Space) |
| **LLaVA-NeXT (LLaMA-3)** | 2-Layer MLP + GELU | **1024** | **4096** (LLaMA-3-8B Space) |
| **Gemma 4 / PaliGemma 2** | Single Linear Layer | **1152** | **2048** / **3072** (Gemma Base Space) |
| **Qwen2-VL** | Cross-Attention Abstractor | **1280** | **3584** (Qwen2-7B Space) |

---

### 3. Proposed Dimension Bridge Techniques (Experiment 1)
When the visual backbone's output dimension ($D_{\text{in}}$) does not match the VLM projector's expected input dimension ($D_{\text{proj}}$), a Dimension Bridge is introduced. We ablate four distinct architectural bridging techniques:

#### A. Bottleneck Linear Layer (Control Baseline)
A single parameter-efficient linear coordinate transformation matrix without non-linearities.
* **Formulation:** $Y = XW$, where $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{proj}}}$.
* **Objective:** Serves as a low-overhead benchmark to ensure the bridge itself is not responsible for downstream classification accuracy gains.

#### B. Symmetrical Low-Rank Bottleneck (LoRA-Style MLP)
Decomposes the dimensionality shift into two low-rank projection matrices separated by a non-linear activation function.
* **Formulation:** $Y = \text{GELU}(XW_{\text{down}})W_{\text{up}}$, where $W_{\text{down}} \in \mathbb{R}^{D_{\text{in}} \times r}$, $W_{\text{up}} \in \mathbb{R}^{r \times D_{\text{proj}}}$, and the internal rank is strictly constrained (e.g., $r = 64$).
* **Objective:** Introduces non-linear manifold mapping while keeping the parameter footprint minimal.

#### C. Spatial Patch Pooling / Convolutional Bridge
Inspired by modern VLM processing pipelines (e.g., Qwen2-VL, Gemma 4), this method projects features while simultaneously regularizing the token sequence.
* **Formulation:** A 2D depthwise separable convolution or a spatial pixel-merge layer mapping $2 \times 2$ neighboring spatial patches, followed by a linear projection layer.
* **Objective:** Evaluates whether preserving localized spatial neighborhoods during token dimension transitions yields superior discriminative feature representations.

#### D. Cross-Attention Resampler Bridge
Decouples both sequence length and channel dimension from the projector using a fixed set of learnable query tokens.
* **Formulation:** $Y = \text{MultiHeadAttention}(Q, K, V)$, where $Q \in \mathbb{R}^{k \times D_{\text{proj}}}$ is a set of constant, learnable queries, and $K, V \in \mathbb{R}^{N \times D_{\text{in}}}$ are mapped directly from the visual backbone.
* **Objective:** Tests a highly robust, non-linear sequence compression strategy inspired by Perceiver-style architectures.

---

### 4. Tensor Mapping Pipeline Grid
The table below illustrates how the combined pipeline transitions structural dimensions from raw visual features to final language token mappings during our fine-tuning experiments:

| Visual Backbone Baseline | $D_{\text{in}}$ | Dimension Bridge Target Mapping | $D_{\text{proj}}$ | VLM Projector Target | $D_{\text{out}}$ (To Classifier Head) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DINOv2-Base** | 768 | Linear / Low-Rank Expansion ($768 \rightarrow 1024$) | 1024 | LLaVA 1.5 | 4096 |
| **Vanilla ViT-Large** | 1024 | *Identity Mapping / No Bridge Required* | 1024 | LLaVA 1.5 | 4096 |
| **Vanilla ViT-Large** | 1024 | Linear Expansion ($1024 \rightarrow 1152$) | 1152 | Gemma 4 | 3072 |
| **EVA-02-Large** | 1024 | Spatial Patch-Merge Bridge ($1024 \rightarrow 1280$) | 1280 | Qwen2-VL | 3584 |
| **Vanilla ViT-Huge** | 1280 | *Identity Mapping / No Bridge Required* | 1280 | Qwen2-VL | 3584 |