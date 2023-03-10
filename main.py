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

def expected_multi_VT_size(syndromes: List[int], n:int) -> int:
    S = 0
    for syndrome in syndromes:
        VT_size = calc_VT_size(syndrome, n)
        VT_cover = VT_size*(n+2)
        S = S + VT_cover - (S*VT_cover)//2**(n+1)
    return S

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

def get_total_VT_coverage(code: Code, chosen: List[int]=[]) -> int:
    """
    returns the number of words in code, covered by the chosen VT codes,
    specified by the syndrome
    """
    syndromes = find_code_deletion_ball_syndromes(code)

    exclusion_filter = np.all(np.isin(syndromes, np.array(chosen), invert=True), axis=1)
    return code.shape[0] - syndromes[exclusion_filter].shape[0]

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

    every sequence of n+2 words in the code is the insertion ball of a single word
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

    # reshape the result of the insertion into words of length n+1
    return inserted.reshape((-1, length+1))

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

def count_redundant(new_code: int, existing_codes: List[int], n: int) -> int:
    """
    counts the number of words in the VT code with syndrome `new code`,
    have insertion balls that are covered completely,
    by the codes with syndromes `existing_codes`
    """
    all_words_n = get_all_words(n)
    syndromes = compute_syndrome(all_words_n)
    VTa_filter = syndromes == new_code
    VTa_code = all_words_n[VTa_filter]

    VTa_insertion_ball = find_code_insertion_ball(VTa_code)
    insertion_ball_syndromes = find_code_deletion_ball_syndromes(VTa_insertion_ball)

    words_covered = np.any(np.isin(insertion_ball_syndromes, np.array(existing_codes)), axis=1)
    balls_covered = np.all(words_covered.reshape((-1, n+2)), axis=1)
    return np.count_nonzero(balls_covered)

def create_graphs_remaining_VT_count(n, cosets, added):
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

def create_graphs_percentage_VT_count(n, cosets, added):
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

def create_graphs_remaining_forward_VT(n, cosets, added):
    remaining = 2**(n+1) - np.cumsum(added)
    plt.xticks(cosets)
    plt.plot(cosets, remaining, 'b')
    plt.bar(cosets, remaining, color='maroon', width = 0.4)
    plt.title(f'words uncovered by each VT codes for n={n}')
    plt.xlabel('VT codes added')
    plt.ylabel('remaining uncovered words')
    plt.savefig(f'RemainingWordsFor{n}_ForwardVT.png')
    plt.clf()

def create_graphs_percentage_forward_VT(n, cosets, added):
    percentage = (added) / (2 ** (n + 1))
    plt.xticks(cosets)
    plt.plot(cosets, percentage, 'b')
    plt.bar(cosets, percentage, color='maroon', width=0.4)
    plt.title(f'Parentage of words covered by each VT codes for n={n}')
    plt.xlabel('VT codes added')
    plt.ylabel('Parentage of words covered')
    plt.savefig(f'ParentageWordsFor{n}_ForwardVT.png')
    plt.clf()

def create_graphs_covariance(n, a, VT_codes, cov):
    plt.xticks(VT_codes)
    plt.bar(VT_codes, cov, color='maroon', width=0.4)
    plt.title(f'covariance of words covered by each VT codes for n={n} with VT code {a}')
    plt.xlabel('VT codes')
    plt.ylabel('covariance')
    plt.savefig(f'covariance/CovarianceVTcodes_{n}_with_{a}.png')
    plt.clf()

def calc_covariance(n: int):
    all_words_np = get_all_words(n+1)
    syndromes = find_code_deletion_ball_syndromes(all_words_np)
    VT_codes = np.arange(n+1)

    for a in VT_codes:
        VTa_ind = np.any(syndromes == a, axis=1).astype(np.float_)
        VTa_mean = np.mean(VTa_ind)
        VTa_normal = VTa_ind - VTa_mean

        VT_codes_sans_a = np.delete(VT_codes, a)
        cov = np.zeros_like(VT_codes_sans_a, dtype=np.float_)

        for i, b in enumerate(VT_codes_sans_a):
            VTb_ind = np.any(syndromes == b, axis=1).astype(np.float_)
            VTb_mean = np.mean(VTb_ind)
            VTb_normal = VTb_ind - VTb_mean

            cov[i] = np.mean(VTa_normal*VTb_normal)

        create_graphs_covariance(n, a, VT_codes_sans_a, cov)

n = 15

cosets, counts = collect_coset_coverage(n)

cosets_so_far = []
total_redundant = 0
for coset in cosets:
    redundant = count_redundant(coset, cosets_so_far, n)
    print(f"adding {redundant} redundant words, after {len(cosets_so_far)} VT codes added")
    total_redundant += redundant
    cosets_so_far += [coset]

print(f"the overall number of redundant words in {total_redundant}")
print(f"the total number of words in the code is {calc_VT_size(0, n)*cosets.size}")

"""
n = 7
a = 0

all_words_n = get_all_words(n)
syndromes = compute_syndrome(all_words_n)
VTa_filter = syndromes == a
VTa_code = all_words_n[VTa_filter]

VTa_insertion_ball = find_code_insertion_ball(VTa_code)
insertion_ball_syndromes = find_code_deletion_ball_syndromes(VTa_insertion_ball)

ball_per_word = insertion_ball_syndromes
sorted_ball_per_word = np.sort(ball_per_word, axis=1)



for arr in np.array_split(sorted_ball_per_word, n+2):
    chosen_syndromes = []
    while arr.size > 0:
        max_syndrome = -1
        max_count = 0
        for i in range(1, n+1):
            u, c = np.unique(unique_per_word(arr) == i, return_counts=True)
            synd_count = c[u]
            if (synd_count.size > 0 and max_count < synd_count):
                max_count = synd_count
                max_syndrome = i
        arr = arr[np.all(arr != max_syndrome, axis=1)]
        chosen_syndromes += [max_syndrome]
    print(chosen_syndromes)
"""