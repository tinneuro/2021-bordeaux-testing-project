import pytest
import logistic
import math

@pytest.mark.parametrize(["x","r","expected"], [(0.1,2.2, 0.198)])
def test_logistic_map(x,r,expected):
	result = logistic.logistic_map(x,r)
	assert math.isclose(result,expected)
	

