"""
--------------------------
Created Tue Feb 08 2022
github.com/marcodeangelis
MIT License
--------------------------
"""
from __future__ import annotations
from typing import (Sequence, Sized, Iterable, Optional, Any, Tuple, Union)
# Sequence: # Must have __len__() and __getitem__(). Ex.: Tuple, List, Range
# Sized: # It suffices to have len() 
# Iterable: # Must have __iter__() and __next__(). Ex.: Dict, Set, Tuple, List, numpy.array
# from intervals.methods import (lo,hi,width,rad,mag,straddlezero,isinterval)

import numpy
from numpy import (ndarray,asarray,stack,transpose,ascontiguousarray)
# float32=numpy.float32

from intervals.arithmetic import (multiply,divide)

MACHINE_EPS = 7./3 - 4./3 - 1

NUMERIC_TYPES =     {'int','float','complex',                   # Python numbers     
                    'int8','int16','int32','int64','intp',      # Numpy integers
                    'uint8','uint16','uint32','uint64','uintp', # Numpy unsigned integers
                    'float16','float32','float64','float_',     # Numpy floats and doubles
                    'complex64','complex128','complex_'}        # Numpy complex floats and doubles

# INTEGERS =          {'int','int8','int16','int32','int64','intp','uint8','uint16','uint32','uint64','uintp'}
# FLOATS =            {'float','float16','float32','float64','float_'}


def show(x: Interval) -> str:
    if len(x)==0: return f'[{x.lo},{x.hi}]' 
    elif len(x.shape)==1: return '\n'.join([f'[{xi.lo},{xi.hi}]' for xi in x]) # vector of intervals
    elif len(x.shape)==2: 
        n,d = x.shape
        return '\n'.join([' '.join([f'{xi.val}' for xi in x[i,:]]) for i in range(n)]) # matrix of intervals
    else: return f'{x.val}'

class Interval():
    """
    --------------------------
    Created Feb 2022
    github.com/marcodeangelis
    MIT License
    --------------------------

    Interval is the main class. 

    """
    def __repr__(self): # return
        return show(self)
    def __str__(self): # print
        return show(self)
    def __init__(self,
                 lo: Union[float,ndarray], 
                 hi: Optional[Union[float,ndarray]] = None) -> None:
        if hi is None:
            hi = lo.copy()
        self.__unsized = True
        self.__lo = asarray(lo, dtype=float)
        self.__hi = asarray(hi, dtype=float) # check lo and hi have same shape
        # if (len(self.__hi.shape)>0) | (len(self.__hi.shape)>0): self.__unsized = False
        self.__shape = self.__lo.shape
        # self.__scalar = (self.__shape==()) | (self.__shape==(1,))
    def __len__(self):
        if self.unsized: return 0 # interval object is not sized, perhaps return an error: TypeError: len() of unsized object
        else: return self.__lo.shape[0] 
    def __iter__(self): # https://realpython.com/introduction-to-python-generators/
        lo_iter, hi_iter = numpy.nditer(self.lo),numpy.nditer(self.hi)
        while True:
            try: yield Interval(lo=next(lo_iter),hi=next(hi_iter))
            except StopIteration: break
        pass
    def __next__(self):
        pass
    def __getitem__(self, i: Union[int, slice]): # make class indexable
        return Interval(lo=self.__lo[i],hi=self.__hi[i])
    # -------------- METHODS -------------- #
    @property
    def lo(self) -> ndarray:
        return self.__lo # return transpose(transpose(self.__val)[0]) # from shape (3,7,2) to (2,7,3) to (3,7)
    @property
    def hi(self) -> ndarray:
        return self.__hi # return transpose(transpose(self.__val)[1])
    @property
    def unsized(self):
        if (len(self.__hi.shape)>0) | (len(self.__hi.shape)>0): return False
        else: return True
    @property
    def val(self):
        if self.unsized: return asarray([self.__lo,self.__hi],dtype=float)
        else: return transpose(stack((self.__lo,self.__hi)))
    @property
    def scalar(self):
        return (self.shape==()) | (self.shape==(1,))
    @property
    def shape(self):
        return self.__shape
    # -------------- ARITHMETIC -------------- #
    def __add__(self,other):
        otherType = other.__class__.__name__
        if (otherType == 'ndarray') | (otherType in NUMERIC_TYPES): lo, hi = self.lo + other, self.hi + other
        elif otherType == 'Interval': lo,hi = self.lo + other.lo, self.hi + other.hi
        else: return NotImplemented # TypeError: unsupported operand type(s) for +: 'int' and 'Interval' (for example)
        return Interval(lo,hi)
    def __radd__(self, left):
        leftType = left.__class__.__name__
        if (leftType == 'ndarray') | (leftType in NUMERIC_TYPES): return self.__add__(left)
        else: return NotImplemented # TypeError: unsupported operand type(s) for +: 'int' and 'Interval' (for example)
    def __sub__(self, other):
        otherType = other.__class__.__name__
        if (otherType == 'ndarray') | (otherType in NUMERIC_TYPES): lo,hi = self.lo - other, self.hi - other
        elif otherType == 'Interval': lo, hi = self.lo - other.hi, self.hi - other.lo
        else: NotImplemented
        return Interval(lo,hi)
    def __rsub__(self, left):
        leftType = left.__class__.__name__
        if (leftType == 'ndarray') | (leftType in NUMERIC_TYPES): lo, hi = left - self.hi, left - self.lo
        else: return NotImplemented #print("Error: not among the allowed types.")
        return Interval(lo,hi)
    def __mul__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            if other >= 0: lo, hi = self.lo * other, self.hi * other
            else: lo, hi = self.hi * other, self.lo * other
        elif otherType == 'ndarray': # check self and other have same shape
            lo,hi = numpy.empty(self.__lo.shape),numpy.empty(self.__lo.shape)
            if len(other.shape)==0: self.__mul__(float(other)) # safety net for ndarrays with no shape
            other_positive = other >= 0
            other_negative = other_positive==False
            lo[other_positive]=self.lo[other_positive] * other[other_positive]
            hi[other_positive]=self.hi[other_positive] * other[other_positive]
            lo[other_negative]=self.hi[other_negative] * other[other_negative]
            hi[other_negative]=self.lo[other_negative] * other[other_negative]
        elif otherType == 'Interval':
            lo,hi = multiply(self,other)
        else: return NotImplemented
        return Interval(lo,hi)
    def __rmul__(self, left):
        leftType = left.__class__.__name__
        if (leftType == 'ndarray') | (leftType in NUMERIC_TYPES): return self.__mul__(left)
        else: return NotImplemented
    def __truediv__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            if other == 0: raise ZeroDivisionError
            if other > 0: lo, hi = self.lo / other, self.hi / other
            else: lo, hi = self.hi / other, self.lo / other
        elif otherType == 'ndarray':
            lo,hi = numpy.empty(self.__lo.shape),numpy.empty(self.__lo.shape)
            if any(other.flatten()==0): raise ZeroDivisionError
            other_positive = other > 0
            other_negative = other_positive==False
            lo[other_positive]=self.lo[other_positive] / other[other_positive]
            hi[other_positive]=self.hi[other_positive] / other[other_positive]
            lo[other_negative]=self.hi[other_negative] / other[other_negative]
            hi[other_negative]=self.lo[other_negative] / other[other_negative]
            pass
        elif otherType == 'Interval':
            lo,hi = divide(self,other)
        else: NotImplemented
        return Interval(lo,hi)
    def __rtruediv__(self, left):
        leftType = left.__class__.__name__
        # lo,hi = numpy.empty(self.__lo.shape),numpy.empty(self.__hi.shape)
        self_lo, self_hi = self.lo, self.hi
        self_straddle_zero = numpy.any((self_lo.flatten()<=0) & (self_hi.flatten()>=0))
        if self_straddle_zero: raise ZeroDivisionError
        if (leftType == 'ndarray') | (leftType in NUMERIC_TYPES): 
            if left >= 0: lo, hi = left / self_hi, left / self_lo
            else: lo, hi = left / self_lo, left / self_hi
        else: return NotImplemented
        return Interval(lo,hi)

# def iterator(x:Interval) -> Interval:
#     lo_iter,hi_iter = numpy.nditer(x.lo()),numpy.nditer(x.hi())
#     while True: yield Interval(lo=next(lo_iter),hi=next(hi_iter))


def is_Interval(x:Any) -> bool:
    x_class_name = x.__class__.__name__
    return x_class_name == 'Interval'