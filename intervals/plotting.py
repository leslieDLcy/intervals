from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from intervals.number import Interval

"""
--------------------------
Created Feb 2022
Marco De Angelis
github.com/marcodeangelis
Editted by Leslie Feb 2024 
MIT License
--------------------------
"""

def plot_intervals(x, y_i, **kwargs):
    """ plot intervals vertically 
    
    args:
        x: array-like
            x-axis locations
        y_i: array-like
            array of intervals
    """
    
    fig, ax = plt.subplots()
    if len(x.shape) > 1:
        for xx, interval in zip(x, y_i):
            ax.plot([xx, xx], [interval.hi, interval.lo], 'r', **kwargs)
    else:
        ax.plot([x, x], [y_i.hi, y_i.lo], 'r', **kwargs)


def plot_lower_bound(x, y_i, **kwargs):
    """ plot lower bound of intervals 
    
    args:
        x: array-like
            x-axis locations
        y_i: array-like
            array of intervals
    """
    
    fig, ax = plt.subplots()
    ax.scatter(x, y_i.lo, **kwargs)


