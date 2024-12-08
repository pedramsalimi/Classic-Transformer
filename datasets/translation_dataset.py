import torch
from torch.utils.data import Dataset
from utils.vocab import Vocab

class TranslationDataset(Dataset):
    def __init__(self, file_path, max_len=20):
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Each line: "english \t french"
        pairs = []
        for line in lines:
            if "\t" in line:
                eng, fra = line.strip().split("\t")
                pairs.append((eng.lower(), fra.lower()))
        
        # Build vocab
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()

        # Collect all English and French sentences for vocab building
        eng_sentences = [p[0] for p in pairs]
        fra_sentences = [p[1] for p in pairs]

        self.src_vocab.build_vocab(eng_sentences)
        self.tgt_vocab.build_vocab(fra_sentences)

        self.max_len = max_len
        self.data = []
        # Encode each pair
        for eng, fra in pairs:
            src_ids = self.src_vocab.encode(eng)
            tgt_ids = self.tgt_vocab.encode(fra)

            tgt_in = tgt_ids[:-1]
            tgt_out = tgt_ids[1:]

            # Truncate/pad sequences to max_len
            src_ids = src_ids[:self.max_len]
            tgt_in = tgt_in[:self.max_len]
            tgt_out = tgt_out[:self.max_len]

            src_ids = src_ids + [self.src_vocab.token_to_idx["<pad>"]] * (self.max_len - len(src_ids))
            tgt_in = tgt_in + [self.tgt_vocab.token_to_idx["<pad>"]] * (self.max_len - len(tgt_in))
            tgt_out = tgt_out + [self.tgt_vocab.token_to_idx["<pad>"]] * (self.max_len - len(tgt_out))

            self.data.append((src_ids, tgt_in, tgt_out))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_in, tgt_out = self.data[idx]
        return torch.tensor(src_ids), torch.tensor(tgt_in), torch.tensor(tgt_out)

if __name__=="__main__":
    dataset = TranslationDataset("data/eng_fra.txt", max_len=64)