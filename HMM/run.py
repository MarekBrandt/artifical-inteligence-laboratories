from hmm import HMM
from utils import load_data, evaluate

if __name__ == "__main__":
    hidden, data = load_data()
    hmm_obj = HMM(*data)
    evaluate(hmm_obj.forward()[0], hidden, "Forward algorithm")
    evaluate(hmm_obj.backward()[0], hidden, "Backward algorithm")
    evaluate(hmm_obj.forward_backward(), hidden, "Forward-backward algorithm")
    evaluate(hmm_obj.viterbi(), hidden, "Viterbi")
