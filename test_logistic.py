import pytest
import logistic
import math
import numpy as np

@pytest.mark.implementation
@pytest.mark.parametrize(["x","r","expected"], [(0.1,2.2, 0.198)])
def test_logistic_map(x,r,expected):
	result = logistic.logistic_map(x,r)
	assert math.isclose(result,expected)
	
@pytest.mark.behavior
def test_converge():
	itr = 100
	r = 1.5

	for ii in range(100):
		
		x = np.random.rand(1)
		assert math.isclose(logistic.logistic_iterate(itr, x, r)[-1],1/3)
	

