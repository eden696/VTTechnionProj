import numpy as np
from numpy.typing import NDArray
import itertools
import sys
from typing import List, Tuple
from sympy.ntheory.factor_ import totient
from sympy.ntheory import mobius
from sympy import gcd
import matplotlib.pyplot as plt

Word = NDArray[np.uint8]    # 1 dimentional binary array
Code = NDArray[np.uint8]    # 2 dimentional binary array,
                            # where each row represents a word in the code

np.set_printoptions(threshold=sys.maxsize)


def calc_VT_size(a: int, n: int) -> int:
    sum_ = 0

    for d in [x for x in range(1, n+2, 2) if ((n+1) % x == 0)]:
        exp = 2**((n+1)//d)
        gcd_ = d//gcd(d, a)
        expr = totient(d)*mobius(gcd_)//totient(gcd_)
        sum_ += exp*expr

    return sum_//(2*(n+1))

def compute_syndrome(code: Code) -> Word:
    """
    compute the VT code syndrome for each element in the code
    """
    n = code.shape[1]
    return np.mod(np.sum((1+np.arange(n))*code, axis=1),n+1)

def find_code_deletion_ball_syndromes(code: Code) -> Code:
    """
    compute the syndromes of deletion ball for each word in the code,
    i.e. words that are created by removing a single bit
    """
    length = code.shape[1]
    word_count = code.shape[0]
    indicies = np.arange(length)*length + np.arange(length)

    # create n instances of each word,
    # so that the deletions can be done independantly for each one
    repeated = np.repeat(code, length, axis=0)
    # flatten all instances of each word,
    # so that insertions can be performed on uniform indicies
    flattened = repeated.reshape((word_count, -1))

    deleted = np.delete(flattened, indicies, axis=1).reshape((-1, length-1))

    # get the all of the syndromes for each word
    syndromes = compute_syndrome(deleted).reshape((word_count, -1))
    # sort values for visual clarity
    return np.sort(syndromes, axis=1)

def unique_per_word(code: Code) -> Word:
    """
    return a flattened array,
    where each element in a word appears once per word.

    elements must be sorted per word
    """
    length = code.shape[1]
    word_count = code.shape[0]
    flattened = code.reshape(-1)
    # select only elements whose neighbor is not identical to them
    # this reduces every run of a single element into just that element
    selection = np.ones(flattened.size, dtype=np.bool_)
    selection[1:] = flattened[1:] != flattened[:-1]
    # ensure duplicates are not deleted accross word boundaries
    selection[np.arange(word_count)*length] = True
    return flattened[selection]

def get_coverage_per_VT(code: Code, exclude: List=[]) -> NDArray[np.int_]:
    """
    count the number of words in the code, which are covered by each VT code,
    and place the number of VTa(n) at index a

    does not count words covered by syndromes in the exclude list
    """
    n = code.shape[1]
    total_counts = np.zeros(n, dtype=np.int_)

    # find the syndromes of words in the deletion ball of each word
    syndromes = find_code_deletion_ball_syndromes(code)

    # remove words where one of the syndromes is in the excluded list
    exclution_filter = np.all(np.isin(syndromes, np.array(exclude), invert=True), axis=1)
    filtered_syndromes = syndromes[exclution_filter]

    # count the number of instances per word and add to the total
    cosets, counts = np.unique(unique_per_word(filtered_syndromes), return_counts=True)
    total_counts[cosets] += counts

    return total_counts

def get_all_words(n: int) -> Code:
    """
    return every word of length n
    """
    all_combinations = itertools.product([0, 1], repeat=n)
    flattened = itertools.chain.from_iterable(all_combinations)
    array = np.fromiter(flattened, dtype=np.uint8, count=(n*2**n))
    return array.reshape(-1, n)

def collect_coset_coverage(n: int) -> Tuple[Word, Word]:
    """
    returns the cosets used during the greedy algorithm,
    and the number of additional words covered in each round
    """

    all_words_np = get_all_words(n+1)
    cosets = []
    counts = []

    while True:
        coverage_per_VT = get_coverage_per_VT(all_words_np, cosets)

        next_coset = np.argmax(coverage_per_VT)
        next_count = coverage_per_VT[next_coset]

        if next_count <= 0:
            break

        cosets += [next_coset]
        counts += [coverage_per_VT[next_coset]]

    return np.array(cosets), np.array(counts)

def creat_graphs_remaining_VT_count(n, cosets, added):
    VT_count = np.arange(cosets.size, dtype=np.int_)
    remaining = 2**(n+1) - np.cumsum(added)
    plt.xticks(VT_count)
    plt.plot(VT_count, remaining, 'b')
    plt.bar(VT_count, remaining, color='maroon', width = 0.4)
    plt.title(f'words uncovered by VT codes for n={n}')
    plt.xlabel('Num of VT codes')
    plt.ylabel('remaining uncovered words')
    plt.savefig(f'RemainingWordsFor{n}.png')
    plt.clf()

def creat_graphs_percentage_VT_count(n, cosets, added):
    VT_count = np.arange(cosets.size, dtype=np.int_)
    percentage = (added) / (2 ** (n + 1))
    plt.xticks(VT_count)
    plt.plot(VT_count, percentage, 'b')
    plt.bar(VT_count, percentage, color='maroon', width=0.4)
    plt.title(f'Parentage of words covered by added VT codes for n={n}')
    plt.xlabel('Num of VT codes')
    plt.ylabel('Parentage of words covered')
    plt.savefig(f'ParentageWordsFor{n}.png')
    plt.clf()

def creat_graphs_remaining_forward_VT(n, cosets, added):
    remaining = 2**(n+1) - np.cumsum(added)
    plt.xticks(cosets)
    plt.plot(cosets, remaining, 'b')
    plt.bar(cosets, remaining, color='maroon', width = 0.4)
    plt.title(f'words uncovered by each VT codes for n={n}')
    plt.xlabel('VT codes added')
    plt.ylabel('remaining uncovered words')
    plt.savefig(f'RemainingWordsFor{n}_ForwardVT.png')
    plt.clf()

def creat_graphs_percentage_forward_VT(n, cosets, added):
    percentage = (added) / (2 ** (n + 1))
    plt.xticks(cosets)
    plt.plot(cosets, percentage, 'b')
    plt.bar(cosets, percentage, color='maroon', width=0.4)
    plt.title(f'Parentage of words covered by each VT codes for n={n}')
    plt.xlabel('VT codes added')
    plt.ylabel('Parentage of words covered')
    plt.savefig(f'ParentageWordsFor{n}_ForwardVT.png')
    plt.clf()
