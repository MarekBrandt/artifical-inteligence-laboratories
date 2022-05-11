import numpy as np

def load_data():
    np.random.seed(123)
    seq_len=30
    transition = [[0.4, 0.2, 0.0, 0.4],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.5, 0.0, 0.0, 0.5],
                  [0.4, 0.2, 0.0, 0.4]]
    emission = [[0.35, 0.45, 0.1, 0.05],
                [0.3 , 0.4 , 0.2, 0.1 ],
                [0.7 , 0.2 , 0.1, 0.0 ],
                [0.1 , 0.3 , 0.4, 0.2 ]]
    initial =   [5/14, 1/7, 1/7, 5/14]
    hidden =    [0,3,0,3,0,1,2,3,3,0,0,0,1,2,0,3,3,0,3,1,2,3,3,0,3,0,1,2,3,3,0,0,3,1,2,3,3,3,0,0,1,2]
    observed =  [1,1,1,2,1,2,0,3,2,1,2,1,1,0,0,2,1,0,2,0,0,3,1,0,2,1,0,1,1,2,0,1,1,1,0,3,2,1,0,1,0,1]

    return hidden, (observed, transition, emission, initial)

def generate_data(seed):
    np.random.seed(seed)
    hidden_dim, observe_dim, seq_len = np.random.randint(3, 30, 3)
    # init matrices with random numbers
    transition = np.random.random((hidden_dim, hidden_dim))
    emission = np.random.random((hidden_dim, observe_dim))
    initial = np.random.random(hidden_dim)
    # normalize rows
    emission /= emission.sum(axis=1)[:, np.newaxis]
    transition /= transition.sum(axis=1)[:, np.newaxis]
    initial /= initial.sum()

    hidden = generate_hidden(initial, transition, seq_len)
    observed = generate_observed(hidden, emission, seq_len)

    return observed, transition, emission, initial


def generate_hidden(I, A, N):
    hidden = []
    K = len(A)
    hidden.append(np.random.choice(list(range(K)), 1, p=I)[0])
    for _ in range(N-1):
        last_obs = hidden[-1]
        hidden.append(np.random.choice(list(range(K)), 1, p=A[last_obs])[0])
    return np.array(hidden)

def generate_observed(H, B, N):
    observed = []
    M = len(B[0])
    for i in range(N):
        hidden_state = H[i]
        observed.append(np.random.choice(list(range(M)), 1, p=B[hidden_state])[0])
    return np.array(observed)

def evaluate(results, expected, name):
    diff = results == expected
    print(name + ": ")
    print("\tAccuracy: " + str(np.sum(diff) / len(results)))
    print("\tMisclassified positions: " + str(np.where(diff == 0)[0]))
