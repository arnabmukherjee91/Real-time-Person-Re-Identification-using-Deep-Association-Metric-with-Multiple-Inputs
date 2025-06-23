import numpy as np

def cosine_distance(a, b, data_is_normalized= False):
	#Normalize vector to unit length
	if not data_is_normalized:
		a = np.asarray(a) / np.linalg.norm(a, axis= 1, keepdims= True)
		b = np.asarray(b) / np.linalg.norm(b, axis= 1, keepdims= True)

	#cosine distance
	return np.asscalar(1. - np.dot(a, b.T))


def search(a, threshold):
	a = np.array(a)
	tmp = np.where((a <= threshold))[0][:1]
	if(len(tmp) == 0):
		return None
	else:
		return np.asscalar(tmp)
