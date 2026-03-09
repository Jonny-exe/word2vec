import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500) # to avoid oferflow
    return 1 / (1 + np.exp(-x))


class SkipGramNegSampling:

    def __init__(self, vocab_size, embed_dim):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # input embeddings
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01

        # output embeddings
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01

    def forward(self, center, context, negatives):

        v_c = self.W_in[center]
        u_o = self.W_out[context]

        # positive score
        pos_score = sigmoid(np.dot(v_c, u_o))

        # negative scores
        neg_vecs = self.W_out[negatives]
        neg_score = sigmoid(np.dot(neg_vecs, v_c))

        return v_c, u_o, neg_vecs, pos_score, neg_score

    def compute_loss(self, pos_score, neg_score):

        loss_pos = -np.log(pos_score + 1e-10)
        loss_neg = -np.sum(np.log(1 - neg_score + 1e-10))

        return loss_pos + loss_neg

    def backward(self, center, context, negatives, v_c, u_o, neg_vecs, pos_score, neg_score, lr):

        # gradients
        grad_pos = pos_score - 1
        grad_neg = neg_score

        # update context vector
        self.W_out[context] -= lr * grad_pos * v_c

        # update negative vectors
        for i, neg in enumerate(negatives):
            self.W_out[neg] -= lr * grad_neg[i] * v_c

        # update center vector
        grad_center = grad_pos * u_o + np.sum(grad_neg[:, None] * neg_vecs, axis=0)

        num_samples = 1 + len(negatives)
        self.W_in[center] -= (lr / num_samples) * grad_center
