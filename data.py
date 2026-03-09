# data.py
import numpy as np
import random
from collections import Counter

class Word2VecDataset:
    def __init__(self, sentences, window_size=2, neg_samples=5, val_split=0.2, seed=42):
        """
        sentences: list of list of words
        window_size: context window for skip-gram
        neg_samples: number of negative samples per positive
        val_split: fraction of sentences used for validation
        """
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Flatten corpus to build vocab
        all_words = [word for sent in sentences for word in sent]
        self.word_counts = Counter(all_words)
        self.vocab = sorted(self.word_counts.keys())
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.vocab)

        # Encode all sentences
        self.encoded_corpus = [[self.word2idx[w] for w in sent] for sent in sentences]

        # Split sentences for train/val
        split_idx = int(len(sentences) * (1 - val_split))
        self.train_sentences = self.encoded_corpus[:split_idx]
        self.val_sentences = self.encoded_corpus[split_idx:]

        # Compute negative sampling distribution
        freq = np.array([self.word_counts[w] for w in self.vocab], dtype=np.float32)
        freq = freq ** 0.75  # as in Mikolov et al.
        self.neg_distribution = freq / freq.sum()

        # Generate train/val pairs
        self.train_pairs = self.generate_pairs(self.train_sentences)
        self.val_pairs = self.generate_pairs(self.val_sentences)

    def generate_pairs(self, sentences):
        """
        Generate skip-gram pairs (center, context) from encoded sentences
        """
        pairs = []
        for sent in sentences:
            for i, center in enumerate(sent):
                window = random.randint(1, self.window_size)
                # context words in window
                for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                    if j == i:
                        continue
                    context = sent[j]
                    pairs.append((center, context))
        return pairs

    def sample_negatives(self, batch_size=1):
        """
        Sample negative words according to the distribution
        """
        return np.random.choice(self.vocab_size, size=(batch_size, self.neg_samples), p=self.neg_distribution)

    def get_train_batch(self, batch_size):
        """
        Sample a batch of training pairs
        """
        batch = random.sample(self.train_pairs, batch_size)
        centers = [c for c, _ in batch]
        contexts = [ctx for _, ctx in batch]
        negatives = [self.sample_negatives() for _ in range(batch_size)]
        return np.array(centers), np.array(contexts), np.array(negatives)

    def get_val_batch(self, batch_size):
        """
        Sample a batch of validation pairs
        """
        batch = random.sample(self.val_pairs, batch_size)
        centers = [c for c, _ in batch]
        contexts = [ctx for _, ctx in batch]
        negatives = [self.sample_negatives() for _ in range(batch_size)]
        return np.array(centers), np.array(contexts), np.array(negatives)

if __name__ == "__main__":

    with open("text8", "r") as f:
        text = f.read()


    sentence_len = 10
    words = text.split()[:100]  # Use only first 1M words for faster testing
    print(f"Total words: {len(words)}") # 17005207

    corpus = [words[i:i+sentence_len] for i in range(0, len(words) - len(words) % sentence_len, sentence_len)]
    print(corpus)

    # corpus = [
    #     ["the", "cat", "sat", "on", "the", "mat"],
    #     ["the", "dog", "sat", "on", "the", "rug"]
    # ]

    dataset = Word2VecDataset(corpus, window_size, neg_samples)
