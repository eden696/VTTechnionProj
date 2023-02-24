import numpy as np
from numpy.typing import NDArray
import itertools

Word = NDArray[np.uint8]    # 1 dimentional binary array
Code = NDArray[np.uint8]    # 2 dimentional binary array,
                            # where each row represents a word in the code

def compute_syndrome(code: Code) -> Word:
    """
    compute the VT code syndrome for each element in the code
    """
    n = code.shape[1]
    return np.mod(np.sum((1+np.arange(n))*code, axis=1),n+1)

def get_VT_code(a: int, n: int) -> Code:
    """
    return a list of every word in the code VT_a(n)
    """
    all_words = get_all_words(n)
    VT_filter = compute_syndrome(all_words) == a
    # returns all words whose syndrome is a (as per the definition)
    return all_words[VT_filter]

def get_all_words(n: int) -> Code:
    """
    return every word of length n
    """
    all_combinations = itertools.product([0, 1], repeat=n)
    flattened = itertools.chain.from_iterable(all_combinations)
    array = np.fromiter(flattened, dtype=np.uint8, count=(n*2**n))
    return array.reshape(-1, n)

def find_code_insertion_ball(code: Code) -> Code:
    """
    compute the insertion ball for each word in the code,
    i.e. words that are created by inserting a single bit,
    returning all words in the union of the balls
    """
    length = code.shape[1]
    word_count = code.shape[0]
    # for each word, before index idx, insert the inverse bit at idx
    inv_code = 1 - code
    # except at the end where both 0 and 1 can be appended
    appended = np.repeat([[0,1]], word_count, axis=0)
    values = np.append(inv_code, appended, axis=1)

    # the size of an insertion ball for n length word is n+2,
    ball_size = length + 2

    # create n+2 instances of each word,
    # so that the insertions can be done independantly for each one
    repeated = np.repeat(code, ball_size, axis=0)

    # flatten all instances of each word,
    # so that insertions can be performed on uniform indicies
    flattened = repeated.reshape((word_count, -1))

    # each instance except the last inserts at an index greater by 1,
    # than its predecessor. calculate where that would be in the flattened array
    indicies = (np.arange(ball_size)*length) + np.arange(ball_size)
    # the last instance inserts in the same position as the last,
    # so undo the increment
    indicies[-1] = indicies[-1] - 1

    inserted = np.insert(flattened, indicies, values, axis=1)

    # reshape the result of the insertion into words of length n+1,
    # which can have duplicates across different words in the code
    duplicated = inserted.reshape((-1, length+1))

    # deduplicate the above words and return the resulting code
    return np.unique(duplicated, axis=0)

def find_words_not_covered(code1: Code, code2: Code) -> Code:
    """
    find all words in code1, which are not covered by insertion balls in code2.

    code1 must be a superset the insertion balls of code2,
    and each word in code1 must be unique.
    """

    # find the words covered by code2
    covered = find_code_insertion_ball(code2)
    # add the covered words to code1, so that they would be duplicates
    code3 = np.append(code1, covered, axis=0)

    unique, counts = np.unique(code3, return_counts=True, axis=0)
    # identify which elements are unique despite the append, those are not covered
    return unique[counts == 1]

# get words not covered by VT0, and those not covered by both VT0 and VT((n+1)/2)
# and compare the sizes
