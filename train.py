import numpy as np
import random
from data import Word2VecDataset
from model import SkipGramNegSampling

def compute_validation_loss(model, dataset, num_samples=1000):
    """
    Computes loss on a subset of the validation pairs.
    """
    # Use the val_pairs generated inside the dataset object
    val_pairs = dataset.val_pairs
    if len(val_pairs) == 0:
        return 0.0

    num_samples = min(num_samples, len(val_pairs))
    samples = random.sample(val_pairs, num_samples)
    total_loss = 0

    for center, context in samples:
        # Sample negatives using the dataset's distribution
        negatives = dataset.sample_negatives().flatten()

        # Forward pass (inference mode)
        v_c, u_o, neg_vecs, pos_score, neg_score = model.forward(center, context, negatives)

        # Compute loss
        loss = model.compute_loss(pos_score, neg_score)
        total_loss += loss

    return total_loss / num_samples

def train(corpus, embed_dim=100, window_size=2, neg_samples=10, lr=0.0001, epochs=10):
    # Initialize a single dataset; it handles its own internal train/val split
    dataset = Word2VecDataset(corpus, window_size=window_size, neg_samples=neg_samples, val_split=0.2)

    model = SkipGramNegSampling(dataset.vocab_size, embed_dim)

    print(f"Vocab Size: {dataset.vocab_size}")
    print(f"Training pairs: {len(dataset.train_pairs)} | Val pairs: {len(dataset.val_pairs)}")
    print("-" * 30)

    for epoch in range(epochs):
        total_train_loss = 0

        # Shuffle training pairs each epoch for better convergence
        random.shuffle(dataset.train_pairs)

        for center, context in dataset.train_pairs:
            # Get negative samples for this specific pair
            negatives = dataset.sample_negatives().flatten()

            # Forward pass
            v_c, u_o, neg_vecs, pos_score, neg_score = model.forward(center, context, negatives)

            # Compute loss
            loss = model.compute_loss(pos_score, neg_score)
            total_train_loss += loss

            # Backward pass & Weight Update
            model.backward(center, context, negatives, v_c, u_o, neg_vecs, pos_score, neg_score, lr)

        avg_train_loss = total_train_loss / len(dataset.train_pairs)
        avg_val_loss = compute_validation_loss(model, dataset)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model, dataset

if __name__ == "__main__":
    # Load and preprocess text8
    try:
        with open("text8", "r") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: 'text8' file not found. Please ensure it's in the same directory.")
        exit()

    # Hyperparameters
    SENTENCE_LEN = 50
    MAX_WORDS = 800000 # Increased slightly for a better vocab

    words = text.split()[:MAX_WORDS]
    print(f"Total words processed: {len(words)}")

    # Group words into "sentences"
    corpus = [words[i : i + SENTENCE_LEN] for i in range(0, len(words), SENTENCE_LEN)]

    # Start training
    trained_model, word2vec_data = train(corpus, epochs=10, lr=0.003)
    import test
    test.run_tests(trained_model, word2vec_data)
