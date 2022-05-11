import numpy as np

class HMM:
    def __init__(self, observed, transition_matrix, emission_matrix, initial_distribution):
        self.I = initial_distribution
        self.V = np.array(observed)
        self.A = np.array(transition_matrix)
        self.B = np.array(emission_matrix)

        self.K = self.A.shape[0]
        self.N = self.V.shape[0]


    def forward(self):
        alpha = np.zeros((self.N, self.K))
        for j in range(self.K):
            alpha[0, j] = self.I[j] * self.B[j, self.V[0]]
        for t in range(1, self.N):
            for j in range(self.K):
                alpha[t, j] = 0
                for i in range(self.K):
                    alpha[t, j] += alpha[t-1, i] * self.A[i, j]
                alpha[t, j] = alpha[t, j] * self.B[j, self.V[t]]

        return np.argmax(alpha, axis=1), alpha

    def backward(self):
        beta = np.zeros((self.N, self.K))
        # TODO: calculate backward values


        return np.argmax(beta, axis=1), beta

    def forward_backward(self):
        fbv = np.zeros((self.N, self.K))
        # TODO: calculate forward-backward values


        return np.argmax(fbv, axis=1)

    def viterbi(self):
        T1 = np.empty((self.K, self.N))
        T2 = np.empty((self.K, self.N), np.int)
        # TODO: calculate T1 and T2 values


        # TODO: .. and viterbi hidden state sequance
        viterbi = np.empty(self.N, np.int) # placeholder

        return viterbi





