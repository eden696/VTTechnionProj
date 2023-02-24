from typing import List, Set
import numpy as np
from numpy.typing import NDArray

Word = NDArray[np.uint8] # 1 dimentional binary array
Code = NDArray[np.uint8] # 2 dimentional binary array, representing multiple words

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
def WordInsertionBall(word: Word) -> Set[Word]:
    return

# get the insertion ball of all words in the code togather
def CodeInsertionBall(code: List[Word]) -> Set[Word]:
    return

# FCFS

# exhastive search for words not covered by codes
def getWordsNotCovered(words: Set[Word], codes: List[Code]) -> Set[Word]:
    return

# get words not covered by VT0, and those not covered by both VT0 and VT((n+1)/2)
# and compare the sizes
