import numpy as np
from numpy.typing import NDArray
import itertools
import sys
from typing import Tuple

import matplotlib.pyplot as plt

Word = NDArray[np.uint8]    # 1 dimentional binary array
Code = NDArray[np.uint8]    # 2 dimentional binary array,
                            # where each row represents a word in the code

np.set_printoptions(threshold=sys.maxsize)

from sympy.ntheory.factor_ import totient
from sympy.ntheory import mobius

from sympy import gcd

def calc_VT_size(a: int, n: int) -> int:
    sum_ = 0
    for d in [x for x in range(1, n+2, 2) if ((n+1) % x == 0)]:
        #print(f"d: {d}")
        exp = 2**((n+1)//d)
        gcd_ = d//gcd(d, a)
        expr = totient(d)*mobius(gcd_)//totient(gcd_)
        sum_ += exp*expr

        #print(gcd_)
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

def unique_per_word(code: Code) -> Word:
    """
    return a flattened array,
    where each element in a word appears once per word.

    elements must be sorted per word
    """
    length = code.shape[1]
    word_count = code.shape[0]
    flattened = code.reshape(-1)
    selection = np.ones(flattened.size, dtype=np.bool_)
    selection[1:] = flattened[1:] != flattened[:-1]
    # ensure duplicates are not deleted accross word boundaries
    selection[np.arange(word_count)*length] = True
    return flattened[selection]

def find_best_coverage(code: Code) -> Tuple[int, int]:
    """
    returns the a value of the VT_a(n) code,
    covering the most words in the given code,
    and the number of words covered
    """
    deletion_syndromes = find_code_deletion_ball_syndromes(code)
    dedup_per_word = unique_per_word(deletion_syndromes)
    coset, counts = np.unique(dedup_per_word, return_counts=True)
    max_coset_idx = np.argmax(counts)
    return coset[max_coset_idx], counts[max_coset_idx]
def find_coverages(code: Code)-> Code :
    '''
    :param code: the word not covered from n+1 length.
    :return:  (the syndrome, the counts off words in insertion ball related to those syndromes )
    '''
    deletion_syndromes = find_code_deletion_ball_syndromes(code)
    dedup_per_word = unique_per_word(deletion_syndromes)
    coset, counts = np.unique(dedup_per_word, return_counts=True)
    coset = coset.reshape((1, -1))
    counts = counts.reshape((1, -1))
    return np.append(coset, counts, axis=0)



def collect_coset_coverage(n: int) -> Tuple[Word, Word]:
    """
    returns the cosets used during the greedy algorithm,
    and the number of additional words covered in each round
    """

    code = get_all_words(n+1)
    all_words_np = get_all_words(n+1)
    all_words_n = get_all_words(n)
    syndromes = compute_syndrome(all_words_n)
    cosets = []
    counts = []

    while code.size > 0:
        coset, count = find_best_coverage(code)
        cosets = cosets + [coset]
        counts = counts + [count]
        VT_filter = np.zeros_like(syndromes)
        for a in cosets:
            VT_filter = np.logical_or(VT_filter, syndromes == a)

        code = find_words_not_covered(all_words_np,  all_words_n[VT_filter])
        if (code.size != 0):
            print(f"for coset choise : {coset}")
            print(find_coverages(np.array(code)))
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


for n in range(3,17):
    print (f" for n = {n}")
    cosets, added = collect_coset_coverage(n)

    #creat_graphs_percentage_VT_count(n, cosets, added)
    #creat_graphs_percentage_VT_count(n, cosets, added)
    #creat_graphs_percentage_forward_VT(n, cosets, added)
    #creat_graphs_remaining_forward_VT(n, cosets, added)




'''
for each n well want to :
1. the reminder from the n+1 words space.
2. the percentage added by each VT code 
'''