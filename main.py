from typing import List, Set
import numpy as np
from numpy.typing import NDArray

CodeWord = NDArray[np.int_]
Code = List[CodeWord]

# NIV

# return a list of every word in the code VT_a(n)
def VTCodeGenerator(a: int, n: int) -> Code:
    return

# return all words of length n
def AllWord(n: int) -> Set[CodeWord]:
    return

# EDEN

# return every word in the insertion in the insertion ball of `word`.
# i.e. words that are created by inserting a single bit.
def WordInsertionBall(word: CodeWord) -> Set[CodeWord]:
    return

# get the insertion ball of all words in the code togather
def CodeInsertionBall(code: List[CodeWord]) -> Set[CodeWord]:
    return

# FCFS

# exhastive search for words not covered by codes
def getWordsNotCovered(words: Set[CodeWord], codes: List[Code]) -> Set[CodeWord]:
    return

# get words not covered by VT0, and those not covered by both VT0 and VT((n+1)/2)
# and compare the sizes
