import numpy as np # numpy==1.22.3
import pickle
from hmm import HMM
from utils import generate_data
NUM_TESTS = 10

def test_forward():
    with open("correct_forward.pickle", "rb") as file:
        correct_forward = pickle.load(file)

    is_correct = []
    for i in range(NUM_TESTS):
        data = generate_data(i)
        output = HMM(*data).forward()
        is_correct.append(all(output[0] == correct_forward[i]))
    print(f"Forward tests: {sum(is_correct)} / {NUM_TESTS} passed")

def test_forward_backward():
    with open("correct_forward_backward.pickle", "rb") as file:
        correct_forward_backward = pickle.load(file)

    is_correct = []
    for i in range(NUM_TESTS):
        data = generate_data(i)
        output = HMM(*data).forward_backward()
        is_correct.append(all(correct_forward_backward[i] == output))
    print(f"Forward-backward tests: {sum(is_correct)} / {NUM_TESTS} passed")

def test_viterbi():
    with open("correct_viterbi.pickle", "rb") as file:
        correct_viterbi = pickle.load(file)

    is_correct = []
    for i in range(NUM_TESTS):
        data = generate_data(i)
        output = HMM(*data).viterbi()
        is_correct.append(all(output == correct_viterbi[i]))
    print(f"Viterbi tests: {sum(is_correct)} / {NUM_TESTS} passed")

if __name__=="__main__":
    test_forward()
    test_forward_backward()
    test_viterbi()
