from intervals.number import Interval
from intervals.methods import exp


''' a series of popular activation functions for interval arithmetic '''

def sigmoid(x:Interval): return 1/(1 + exp(-x))
def tanh(x:Interval): return (exp(2*x)-1)/(exp(2*x)+1)