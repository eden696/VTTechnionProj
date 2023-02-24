from typing import List, Set
import numpy as np
from numpy.typing import NDArray
import itertools

Word = NDArray[np.uint8] # 1 dimentional binary array
Code = NDArray[np.uint8] # 2 dimentional binary array, representing multiple words

# NIV

def compute_syndrome(code: Code) -> Word:
    """
    compute the VT code syndrome for each element in the code
    """
    n = code.shape[1]
    return np.mod(np.sum((1+np.arange(n))*code, axis=1),n+1)

def VTCodeGenerator(a: int, n: int) -> Code:
    """
    return a list of every word in the code VT_a(n)
    """
    all_words = AllWords(n)
    VT_filter = compute_syndrome(all_words) == a
    # returns all words whose syndrome is a (as per the definition)
    return all_words[VT_filter]

def AllWords(n: int) -> Code:
    """
    return every word of length n
    """
    all_combinations = itertools.product([0, 1], repeat=n)
    flattened = itertools.chain.from_iterable(all_combinations)
    array = np.fromiter(flattened, dtype=np.uint8, count=(n*2**n))
    return array.reshape(-1, n)

# return every word in the insertion in the insertion ball of `word`.
# i.e. words that are created by inserting a single bit.
def WordInsertionBall(word: Word) -> Code:
    length = word.size
    # insert the inverse of word at idx before idx
    inv_word = 1 - word
    # except at the end where both 0 and 1 can be appended
    appended = [0,1]
    values = np.append(inv_word, [0 , 1])

    extra_length = length + len(appended)
    # create enough copies for each insertion
    # reshape word so that it gets repeated, and not its elements
    repeated = np.repeat(word.reshape((-1, length)), extra_length, axis=0)

    # insert flattens the array unless axis is specified,
    # to allow for non-uniform insertions
    indicies = (np.arange(extra_length)*length) + np.arange(extra_length)
    # the last index is to the same position as the last instead of increasing
    indicies[-1] = indicies[-1] - 1
    flattened = np.insert(repeated, indicies, values)
    return flattened.reshape(-1, length+1)

# get the insertion ball of all words in the code togather
def CodeInsertionBall(code: List[Word]) -> Set[Word]:
    return

# FCFS

# exhastive search for words not covered by codes
def getWordsNotCovered(words: Set[Word], codes: List[Code]) -> Set[Word]:
    return

# get words not covered by VT0, and those not covered by both VT0 and VT((n+1)/2)
# and compare the sizes
