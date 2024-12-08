import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import load_config
from models.transformer import Transformer
from utils.training_utils import generate_subsequent_mask
from datasets.translation_dataset import TranslationDataset

# 1. Load config
config = load_config('config.json')

SRC_VOCAB_SIZE = config["SRC_VOCAB_SIZE"]
TGT_VOCAB_SIZE = config["TGT_VOCAB_SIZE"]
EMBED_DIM = config["EMBED_DIM"]
NUM_HEADS = config["NUM_HEADS"]
FF_DIM = config["FF_DIM"]
ENC_LAYERS = config["ENC_LAYERS"]
DEC_LAYERS = config["DEC_LAYERS"]
DROPOUT = config["DROPOUT"]
LR = config["LEARNING_RATE"]
EPOCHS = config["EPOCHS"]
MAX_LEN = config["MAX_LEN"]
BATCH_SIZE = config["BATCH_SIZE"]

# 2. Create dataset and dataloader
dataset = TranslationDataset("data/eng_fra.txt", max_len=MAX_LEN)
src_vocab = dataset.src_vocab
tgt_vocab = dataset.tgt_vocab

# Update VOCAB_SIZE in case actual vocabulary is smaller:
SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialize model, criterion, optimizer
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_encoder_layers=ENC_LAYERS,
    num_decoder_layers=DEC_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
)

criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.token_to_idx["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src, tgt_in, tgt_out = batch
        # src: [B, L], tgt_in: [B, L], tgt_out: [B, L]

        # Create masks:
        src_mask = (src != src_vocab.token_to_idx["<pad>"]).unsqueeze(1).unsqueeze(2)
        # generate_subsequent_mask:
        tgt_sub_mask = generate_subsequent_mask(tgt_in.size(1)).to(src.device)
        tgt_pad_mask = (tgt_in != tgt_vocab.token_to_idx["<pad>"]).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        # Forward pass
        outputs = model(src, tgt_in, src_mask, tgt_mask)
        # outputs: [B, L, TGT_VOCAB_SIZE]

        # Reshape for loss: flatten batch and sequence
        outputs = outputs.view(-1, TGT_VOCAB_SIZE)
        tgt_out = tgt_out.view(-1)

        loss = criterion(outputs, tgt_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")
