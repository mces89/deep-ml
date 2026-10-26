import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	att = np.dot(Q, K.T)/np.sqrt(Q.shape[1])
	att += mask
	att -= att.max(axis=-1, keepdims=True)
	exp_att = np.exp(att)
	score = exp_att/np.sum(exp_att, axis=-1, keepdims=True)
	return np.dot(score, V)
	
