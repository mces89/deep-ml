import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)