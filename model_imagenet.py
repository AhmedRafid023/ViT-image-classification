import torch
import torch.nn as nn
import lightning as L
import gc
from transformers import AutoModel, AutoModelForImageTextToText, LlavaForConditionalGeneration

class ProjectorAblationModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.situation = config['experiment']['situation']
        self.lr = config['trainer'].get('learning_rate', 1e-4) 
        
        model_id = config['model']['id']
        projector_type = config['model']['projector_type']
        
        # --- 1. Load the Base Vision Encoder (VE) ---
        base_model = AutoModel.from_pretrained(model_id)
        
        if hasattr(base_model, "vision_model"):
            self.ve = base_model.vision_model
            print(f"Detected multimodal model ({model_id}). Extracted vision_model only.")
        else:
            self.ve = base_model
            
        ve_dim = getattr(self.ve.config, "hidden_size", 
                         getattr(self.ve.config, "vision_config", 768))
        if not isinstance(ve_dim, int): ve_dim = ve_dim.hidden_size

        # --- 2. Extract Pre-Trained Projectors directly to self.proj ---
        if projector_type == "llava":
            print("Extracting LLaVA 1.5 Projector Weights...")
            vlm = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="cpu")
            self.proj = vlm.multi_modal_projector.float() # Ensures FP32 for GradScaler
            
            proj_in_dim = self.proj.linear_1.in_features
            llm_hidden_dim = self.proj.linear_2.out_features
            del vlm; gc.collect() 

        elif projector_type == "gemma":
            print("Extracting Gemma 3 Projector Weights...")
            vlm = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
            
            self.proj = None
            possible_names = ["multi_modal_projector", "vision_projector", "projector"]
            
            search_bases = [vlm]
            if hasattr(vlm, "model"):
                search_bases.append(vlm.model)
                
            for base in search_bases:
                for name, module in base.named_children():
                    if any(p in name.lower() for p in possible_names):
                        self.proj = module.float() # Ensures FP32 for GradScaler
                        break
                if self.proj is not None:
                    break
                        
            if self.proj is None:
                raise ValueError("Could not automatically locate the Gemma projector.")
            
            proj_in_dim = vlm.config.vision_config.hidden_size
            
            if hasattr(vlm.config, "text_config"):
                llm_hidden_dim = vlm.config.text_config.hidden_size
            else:
                llm_hidden_dim = vlm.config.hidden_size
                
            del vlm; gc.collect() 

        elif projector_type == "none":
            self.proj = nn.Identity()
            proj_in_dim = ve_dim
            llm_hidden_dim = ve_dim
        else:
            raise ValueError("Invalid projector_type. Choose 'llava', 'gemma', or 'none'.")

        # --- 3. Strict Dimension Guard (No Bridge Layer!) ---
        if projector_type != "none" and ve_dim != proj_in_dim:
            raise ValueError(
                f"Dimension mismatch! Vision Encoder outputs {ve_dim}, but the {projector_type} projector expects {proj_in_dim}. "
                f"Because the bridge layer is removed, you MUST use a matching model."
            )

        # --- 4. Route Logic & Dimensions based on Situation ---
        no_proj_situations = ["train_ve_out_ch", "train_ve_ch"]
        self.use_projector = self.situation not in no_proj_situations
        out_dim = llm_hidden_dim if self.use_projector and projector_type != "none" else ve_dim

        # --- 5. Classifier Head (CH) ---
        self.ch = nn.Linear(out_dim, config['model']['num_classes'])
        self.loss_fn = nn.CrossEntropyLoss()

        self._apply_situation_rules()

    def _apply_situation_rules(self):
        # Default: Freeze everything
        for param in self.parameters():
            param.requires_grad = False
            
        # Always unfreeze the classification head
        for param in self.ch.parameters():
            param.requires_grad = True

        # Situation-specific unfreezing (Full Finetuning)
        if self.situation == "train_ve_out_ch":
            pass # VE frozen, Proj bypassed
        elif self.situation == "train_proj_out_ch":
            pass # VE frozen, Proj frozen
        elif self.situation == "train_proj_ch":
            for param in self.proj.parameters():
                param.requires_grad = True
        elif self.situation == "train_ve_ch":
            for param in self.ve.parameters():
                param.requires_grad = True # Proj bypassed
        elif self.situation == "train_all":
            for param in self.ve.parameters():
                param.requires_grad = True
            for param in self.proj.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown situation: {self.situation}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Situation: {self.situation} | Trainable Params: {trainable:,} / {total:,}")

    def forward(self, pixel_values):
        ve_outputs = self.ve(pixel_values=pixel_values)
        
        if self.use_projector:
            features = ve_outputs.last_hidden_state
            features = self.proj(features)
            
            if features.dim() == 3:
                features = features.mean(dim=1)
        else:
            if hasattr(ve_outputs, 'pooler_output') and ve_outputs.pooler_output is not None:
                features = ve_outputs.pooler_output
            else:
                features = ve_outputs.last_hidden_state[:, 0, :]
            
        return self.ch(features)

    def training_step(self, batch, batch_idx):
        logits = self(batch['pixel_values'])
        loss = self.loss_fn(logits, batch['label'])
        acc = (torch.argmax(logits, dim=1) == batch['label']).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['pixel_values'])
        loss = self.loss_fn(logits, batch['label'])
        acc = (torch.argmax(logits, dim=1) == batch['label']).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(trainable_params, lr=self.lr)