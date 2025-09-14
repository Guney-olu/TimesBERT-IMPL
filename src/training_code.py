import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
import wandb
from safetensors.torch import save_file, load_file
from torch.nn.utils.rnn import pad_sequence

class TrainingConfig:
    DATA_PATH = "data"
    VAL_FILE_INDEX = 0
    AD_ID_COLUMN = "ad_id"
    TIME_COLUMN = "date"
    FEATURE_COLUMNS = [
        'spend', 'impressions', 'cpm', 'frequency', 'clicks', 'ctr',
        'actions_omni_purchase', 'purchase_roas_omni_purchase'
    ]
    DOMAIN_COLUMN = "campaign_name"
    MIN_SEQ_LENGTH = 21
    PATCH_LENGTH = 7
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    NUM_HEADS = 8
    OUTPUT_DIR = "timesbert_ad_performance"
    MODEL_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.json")
    FINAL_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    MASK_RATIO = 0.25
    EPOCHS = 150
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_INTERVAL = 10
    WANDB_PROJECT_NAME = "TimesBERT-Pre-training-Ads"

class AdPerformanceDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.samples = self._load_and_preprocess_data()
        self.domain_map = {domain: i for i, domain in enumerate(pd.Series([s['domain'] for s in self.samples]).unique())}
        domain_map_path = os.path.join(self.config.OUTPUT_DIR, "domain_map.json")
        with open(domain_map_path, "w") as f:
            json.dump(self.domain_map, f)

    def _load_and_preprocess_data(self):
        all_files = glob.glob(os.path.join(self.config.DATA_PATH, "*.csv"))
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        df[self.config.TIME_COLUMN] = pd.to_datetime(df[self.config.TIME_COLUMN])
        df = df.sort_values(by=[self.config.AD_ID_COLUMN, self.config.TIME_COLUMN])
        processed_samples = []
        for ad_id, group in tqdm(df.groupby(self.config.AD_ID_COLUMN), desc="Preprocessing Data"):
            if len(group) < self.config.MIN_SEQ_LENGTH:
                continue
            features = group[self.config.FEATURE_COLUMNS].copy().fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            domain = group[self.config.DOMAIN_COLUMN].iloc[0]
            processed_samples.append({"ad_id": ad_id, "data": torch.tensor(scaled_features, dtype=torch.float32), "domain": domain})
        print(f"Loaded and processed {len(processed_samples)} samples.")
        return processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample['data'].clone()
        if random.random() > 0.5 and len(self.samples) > 1:
            rand_idx = random.randint(0, len(self) - 1)
            while rand_idx == idx:
                rand_idx = random.randint(0, len(self) - 1)
            other_sample_data = self.samples[rand_idx]['data']
            variate_to_replace = random.randint(0, data.shape[1] - 1)
            if other_sample_data.shape[0] >= data.shape[0]:
                data[:, variate_to_replace] = other_sample_data[:data.shape[0], variate_to_replace]
            variate_target = torch.tensor(variate_to_replace, dtype=torch.long)
        else:
            variate_target = torch.tensor(data.shape[1], dtype=torch.long)
        domain_label = self.domain_map[sample['domain']]
        domain_target = torch.tensor(domain_label, dtype=torch.long)
        seq_len, num_features = data.shape
        num_patches = seq_len // self.config.PATCH_LENGTH
        if num_patches == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        input_data = data[:num_patches * self.config.PATCH_LENGTH]
        input_patches = input_data.unfold(0, self.config.PATCH_LENGTH, self.config.PATCH_LENGTH).permute(1, 0, 2)
        num_masked_patches = int(self.config.MASK_RATIO * num_patches)
        masked_indices = random.sample(range(num_patches), num_masked_patches)
        masked_input = input_patches.clone()
        masked_input[:, masked_indices, :] = 0
        mpm_target = input_patches[:, masked_indices, :]
        return {"masked_input_patches": masked_input, "mpm_target": mpm_target, "mask_indices": torch.tensor(masked_indices, dtype=torch.long), "variate_target": variate_target, "domain_target": domain_target}

def custom_collate_fn(batch):
    max_patches = max(item['masked_input_patches'].shape[1] for item in batch)
    padded_inputs, attention_masks = [], []
    for item in batch:
        inp = item['masked_input_patches']
        num_patches = inp.shape[1]
        padding_needed = max_patches - num_patches
        padding_tensor = torch.zeros(inp.shape[0], padding_needed, inp.shape[2])
        padded_input = torch.cat([inp, padding_tensor], dim=1)
        padded_inputs.append(padded_input)
        attn_mask = torch.zeros(max_patches, dtype=torch.bool)
        attn_mask[:num_patches] = True
        attention_masks.append(attn_mask)
    mpm_targets = [item['mpm_target'] for item in batch]
    mask_indices = [item['mask_indices'] for item in batch]
    variate_targets = torch.stack([item['variate_target'] for item in batch])
    domain_targets = torch.stack([item['domain_target'] for item in batch])
    return {"padded_input_patches": torch.stack(padded_inputs), "attention_mask": torch.stack(attention_masks), "mpm_targets_list": mpm_targets, "mask_indices_list": mask_indices, "variate_targets": variate_targets, "domain_targets": domain_targets}

class TimesBERTModel(nn.Module):
    def __init__(self, config, num_domains):
        super().__init__()
        self.config = config
        self.num_variates = len(config.FEATURE_COLUMNS)
        self.patch_embed = nn.Linear(config.PATCH_LENGTH, config.HIDDEN_SIZE)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.HIDDEN_SIZE))
        self.var_sep_token = nn.Parameter(torch.randn(1, 1, config.HIDDEN_SIZE))
        max_patches_per_var = 200
        max_len = 1 + (max_patches_per_var + 1) * self.num_variates
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, config.HIDDEN_SIZE))
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.HIDDEN_SIZE, nhead=config.NUM_HEADS, dim_feedforward=config.HIDDEN_SIZE * 4, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)
        self.mpm_head = nn.Linear(config.HIDDEN_SIZE, config.PATCH_LENGTH)
        self.variate_head = nn.Linear(config.HIDDEN_SIZE, self.num_variates + 1)
        self.domain_head = nn.Linear(config.HIDDEN_SIZE, num_domains)

    def forward(self, patch_embeddings, attention_mask):
        batch_size, _, num_patches, _ = patch_embeddings.shape
        token_sequence, full_attention_mask = [], []
        token_sequence.append(self.cls_token.expand(batch_size, -1, -1))
        full_attention_mask.append(torch.ones(batch_size, 1, dtype=torch.bool, device=patch_embeddings.device))
        for i in range(self.num_variates):
            token_sequence.append(patch_embeddings[:, i, :, :])
            full_attention_mask.append(attention_mask)
            token_sequence.append(self.var_sep_token.expand(batch_size, -1, -1))
            full_attention_mask.append(torch.ones(batch_size, 1, dtype=torch.bool, device=patch_embeddings.device))
        flat_sequence = torch.cat(token_sequence, dim=1)
        full_attention_mask = torch.cat(full_attention_mask, dim=1)
        flat_sequence += self.pos_encoder[:, :flat_sequence.size(1), :]
        src_key_padding_mask = ~full_attention_mask
        transformer_output = self.transformer_encoder(flat_sequence, src_key_padding_mask=src_key_padding_mask)
        domain_output = self.domain_head(transformer_output[:, 0, :])
        var_sep_indices = [1 + (num_patches + 1) * (i + 1) - 1 for i in range(self.num_variates)]
        var_sep_outputs = transformer_output[:, var_sep_indices, :]
        variate_output = self.variate_head(var_sep_outputs)
        patch_outputs = []
        for i in range(self.num_variates):
            start_idx = 1 + i * (num_patches + 1)
            end_idx = start_idx + num_patches
            patch_outputs.append(transformer_output[:, start_idx:end_idx, :])
        all_patch_outputs = torch.stack(patch_outputs, dim=1)
        mpm_output = self.mpm_head(all_patch_outputs).permute(0, 2, 3, 1)
        return mpm_output, variate_output, domain_output

def train():
    cfg = TrainingConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    run = wandb.init(project=cfg.WANDB_PROJECT_NAME, config=vars(cfg))
    print(f"Using device: {cfg.DEVICE}")
    print("Loading dataset...")
    dataset = AdPerformanceDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    num_domains = len(dataset.domain_map)
    model = TimesBERTModel(cfg, num_domains)
    print("Saving model configuration...")
    model_config = {"hidden_size": cfg.HIDDEN_SIZE, "num_layers": cfg.NUM_LAYERS, "num_heads": cfg.NUM_HEADS, "patch_length": cfg.PATCH_LENGTH, "num_variates": len(cfg.FEATURE_COLUMNS), "num_domains": num_domains}
    with open(cfg.MODEL_CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=4)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    print("Starting training...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss, total_mpm_loss, total_var_loss, total_dom_loss = 0, 0, 0, 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
            optimizer.zero_grad()
            padded_patches = batch['padded_input_patches'].to(cfg.DEVICE)
            attention_mask = batch['attention_mask'].to(cfg.DEVICE)
            model_to_call = model.module if isinstance(model, nn.DataParallel) else model
            patch_embeddings = model_to_call.patch_embed(padded_patches)
            mpm_pred, var_pred, dom_pred = model(patch_embeddings, attention_mask)
            batch_mpm_preds_gathered, batch_mpm_targets_gathered = [], []
            for i in range(cfg.BATCH_SIZE):
                preds_for_sample = mpm_pred[i][batch['mask_indices_list'][i]]
                targets_for_sample = batch['mpm_targets_list'][i].permute(1,0,2)
                batch_mpm_preds_gathered.append(preds_for_sample)
                batch_mpm_targets_gathered.append(targets_for_sample)
            mpm_pred_flat = torch.cat([p.flatten() for p in batch_mpm_preds_gathered])
            mpm_target_flat = torch.cat([t.flatten() for t in batch_mpm_targets_gathered]).to(cfg.DEVICE)
            loss_mpm = mse_loss(mpm_pred_flat, mpm_target_flat)
            var_target = batch['variate_targets'].to(cfg.DEVICE)
            num_variates = len(cfg.FEATURE_COLUMNS)
            loss_var = ce_loss(var_pred.view(-1, num_variates + 1), var_target.repeat_interleave(num_variates))
            dom_target = batch['domain_targets'].to(cfg.DEVICE)
            loss_dom = ce_loss(dom_pred, dom_target)
            loss = loss_mpm + loss_var + loss_dom
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mpm_loss += loss_mpm.item()
            total_var_loss += loss_var.item()
            total_dom_loss += loss_dom.item()
        avg_loss, avg_mpm_loss, avg_var_loss, avg_dom_loss = total_loss / len(dataloader), total_mpm_loss / len(dataloader), total_var_loss / len(dataloader), total_dom_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | MPM: {avg_mpm_loss:.4f} | Var: {avg_var_loss:.4f} | Dom: {avg_dom_loss:.4f}")
        wandb.log({"avg_loss": avg_loss, "avg_mpm_loss": avg_mpm_loss, "avg_var_loss": avg_var_loss, "avg_dom_loss": avg_dom_loss}, step=epoch)
        if (epoch + 1) % cfg.CHECKPOINT_INTERVAL == 0:
            print(f"\n--- Checkpointing and Evaluating at Epoch {epoch+1} ---")
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.safetensors")
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            save_file(model_to_save.state_dict(), checkpoint_path)
            reconstruction_plot_path = run_inference(checkpoint_path=checkpoint_path, epoch=epoch+1)
            if reconstruction_plot_path:
                wandb.log({f"reconstruction_epoch_{epoch+1}": wandb.Image(reconstruction_plot_path)}, step=epoch)
            print(f"--- End Evaluation for Epoch {epoch+1} ---\n")
    print("Training finished...")
    final_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    save_file(final_model_to_save.state_dict(), cfg.FINAL_MODEL_SAVE_PATH)
    print(f"Final model artifacts saved locally to {cfg.OUTPUT_DIR}")
    artifact = wandb.Artifact(name=f"{cfg.WANDB_PROJECT_NAME}-model", type="model")
    artifact.add_dir(cfg.OUTPUT_DIR)
    run.log_artifact(artifact)
    run.finish()
    print("Run completed successfully.")

def run_inference(checkpoint_path, epoch):
    cfg = TrainingConfig()
    
    with open(cfg.MODEL_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)
    
    model = TimesBERTModel(cfg, model_config['num_domains'])
    
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(cfg.DEVICE)
    model.eval()

    val_file = glob.glob(os.path.join(cfg.DATA_PATH, "*.csv"))[cfg.VAL_FILE_INDEX]
    df = pd.read_csv(val_file)
    df[cfg.TIME_COLUMN] = pd.to_datetime(df[cfg.TIME_COLUMN])
    df = df.sort_values(by=[cfg.AD_ID_COLUMN, cfg.TIME_COLUMN])

    val_group = next((g for _, g in df.groupby(cfg.AD_ID_COLUMN) if len(g) >= cfg.MIN_SEQ_LENGTH), None)
    if val_group is None:
        print("Could not find a suitable ad in the validation file for inference.")
        return None

    ad_id = val_group[cfg.AD_ID_COLUMN].iloc[0]
    features = val_group[cfg.FEATURE_COLUMNS].copy().fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    original_data = torch.tensor(scaled_features, dtype=torch.float32)
    
    seq_len, num_features = original_data.shape
    num_patches = seq_len // cfg.PATCH_LENGTH
    if num_patches == 0: return None
        
    input_data = original_data[:num_patches * cfg.PATCH_LENGTH]
    input_patches = input_data.unfold(0, cfg.PATCH_LENGTH, cfg.PATCH_LENGTH).permute(1, 0, 2)
    
    random.seed(42)
    masked_indices = sorted(random.sample(range(num_patches), int(num_patches * cfg.MASK_RATIO)))
    
    masked_input_patches = input_patches.clone()
    masked_input_patches[:, masked_indices, :] = 0

    model_input_patches = masked_input_patches.unsqueeze(0).to(cfg.DEVICE)
    attn_mask = torch.ones(1, num_patches, dtype=torch.bool, device=cfg.DEVICE)
    
    with torch.no_grad():
        model_to_call = model.module if isinstance(model, nn.DataParallel) else model
        patch_embeddings = model_to_call.patch_embed(model_input_patches)
        mpm_pred, _, _ = model(patch_embeddings, attn_mask)
    
    reconstructed_patches = mpm_pred.cpu().squeeze(0)

    feature_to_plot = random.randint(0, num_features - 1)
    patch_idx_to_plot = random.choice(masked_indices)
    
    original_patch = input_patches[feature_to_plot, patch_idx_to_plot, :].numpy()
    reconstructed_patch = reconstructed_patches[patch_idx_to_plot, :, feature_to_plot].numpy()

    plt.figure(figsize=(12, 6))
    plt.title(f"Epoch {epoch}: Reconstruction for Ad '{ad_id}', Feat '{cfg.FEATURE_COLUMNS[feature_to_plot]}', Patch {patch_idx_to_plot}")
    plt.plot(original_patch, label="Original Data", marker='o')
    plt.plot(reconstructed_patch, label="Reconstructed Data", linestyle='--', marker='x')
    plt.xlabel("Time Step within Patch"); plt.ylabel("Normalized Value")
    plt.legend(); plt.grid(True)
    plot_filename = f"reconstruction_epoch_{epoch}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved reconstruction plot to {plot_filename}")
    return plot_filename

if __name__ == '__main__':
    train()