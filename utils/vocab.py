import collections

class Vocab:
    def __init__(self, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.specials = specials
        self.idx_to_token = []
        self.token_to_idx = {}
        for sp in specials:
            self.add_token(sp)

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def __len__(self):
        return len(self.idx_to_token)

    def build_vocab(self, sentences, max_size=10000):
        # Count frequencies
        counter = collections.Counter()
        for sent in sentences:
            for w in sent.split():
                counter[w] += 1
        # Sort by frequency
        most_common = counter.most_common(max_size - len(self.specials))
        for word, _freq in most_common:
            self.add_token(word)

    def encode(self, sentence):
        # Add <sos> at start and <eos> at end
        tokens = ["<sos>"] + sentence.split() + ["<eos>"]
        return [self.token_to_idx.get(t, self.token_to_idx["<unk>"]) for t in tokens]

    def decode(self, token_ids):
        tokens = []
        for idx in token_ids:
            if idx == self.token_to_idx["<eos>"]:
                break
            if idx == self.token_to_idx["<pad>"]:
                continue
            tokens.append(self.idx_to_token[idx])
        return " ".join(tokens)
