import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def get_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def find_intruder(word_list, model, dataset):
    """Identifies which word doesn't belong in a group based on vector distance."""
    indices = [dataset.word2idx[w] for w in word_list if w in dataset.word2idx]
    if len(indices) < len(word_list):
        print(f"Skipping intruder test: Some words not in vocab.")
        return

    vectors = model.W_in[indices]
    mean_vector = np.mean(vectors, axis=0)

    # Calculate similarity of each word to the group average
    similarities = [get_cosine_similarity(v, mean_vector) for v in vectors]
    intruder_idx = np.argmin(similarities)

    print(f"\nIntruder Test: {word_list}")
    print(f"  -> Predicted Intruder: '{word_list[intruder_idx]}' (Sim to group: {similarities[intruder_idx]:.4f})")

def plot_similarity_heatmap(words, model, dataset):
    """Generates a visual matrix showing how words cluster together."""
    valid_words = [w for w in words if w in dataset.word2idx]
    indices = [dataset.word2idx[w] for w in valid_words]
    vectors = model.W_in[indices]

    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)
    sim_matrix = np.dot(vectors_norm, vectors_norm.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=valid_words, yticklabels=valid_words)
    plt.title("Word Relationship Heatmap")
    plt.show()

def run_arithmetic(plus, minus, add, model, dataset):
    """ (king - man) + woman logic """
    if not all(w in dataset.word2idx for w in [plus, minus, add]):
        return

    vec = model.W_in[dataset.word2idx[plus]] - model.W_in[dataset.word2idx[minus]] + model.W_in[dataset.word2idx[add]]

    # Simple brute force search for top 3
    scores = []
    for i in range(model.vocab_size):
        if dataset.idx2word[i] in [plus, minus, add]: continue
        scores.append((dataset.idx2word[i], get_cosine_similarity(vec, model.W_in[i])))

    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Arithmetic: ({plus} - {minus}) + {add} = {scores[0][0]} ({scores[0][1]:.4f})")

def run_tests(model, dataset):
    print("\n" + "="*50)
    print("🚀 STARTING IMPROVED EMBEDDING SUITE")
    print("="*50)

    # 1. Categorical Logic (Intruder)
    # Testing if model knows 'pizza' is not a number
    find_intruder(["one", "two", "three", "pizza"], model, dataset)
    # Testing if 'war' is not a small number
    find_intruder(["seven", "eight", "nine", "war"], model, dataset)

    # 2. Vector Arithmetic
    print("\n>>> Vector Logic:")
    run_arithmetic("seven", "six", "eight", model, dataset) # Expect 'nine'
    run_arithmetic("history", "war", "peace", model, dataset)
    run_arithmetic("father", "man", "woman", model, dataset)

    # 3. Heatmap Visualization
    print("\n>>> Generating Heatmap...")
    # Choose words likely to be related and some unrelated
    heatmap_words = ["one", "two", "three", "four", "american", "british", "french", "war", "battle"]
    plot_similarity_heatmap(heatmap_words, model, dataset)


    # 4. t-SNE Global Map
    print("\n>>> Generating t-SNE Map...")
    # Select top 300 words for the map
    num_to_plot = min(300, model.vocab_size)
    vectors = model.W_in[:num_to_plot]
    labels = [dataset.idx2word[i] for i in range(num_to_plot)]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_to_plot-1))
    coords = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
    for i, label in enumerate(labels):
        plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=8)
    plt.title("t-SNE Global Word Clusters")
    plt.show()
