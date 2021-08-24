import numpy as np 
def logistic_map(x,r):
	f = r*x*(1-x)
	return f

def logistic_iterate(itr, x, r):
	output = np.zeros(itr+1)
	output[0] = x
	for i in range(itr):
		output[i+1] = logistic_map(output[i],r) 

	return output
