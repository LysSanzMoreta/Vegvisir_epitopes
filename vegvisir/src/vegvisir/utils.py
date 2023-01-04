"""
=======================
2022-2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import argparse
import ast,warnings
import Bio.Align
import numpy as np
from collections import defaultdict
def str2bool(v):
    """Converts a string into a boolean, useful for boolean arguments
    :param str v"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2None(v):
    """Converts a string into None
    :param str v"""

    if v.lower() in ('None'):
        return None
    else:
        v = ast.literal_eval(v)
        return v
def aminoacid_names_dict(aa_types,zero_characters = []):
    """ Returns an aminoacid associated to a integer value
    All of these values are mapped to 0:
        # means empty value/padding
        - means gap in an alignment
        * means stop codon
    :param int aa_types: amino acid probabilities, this number correlates to the number of different aa types in the input alignment
    :param list : character(s) to be set to 0
    """
    if aa_types == 20:
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20}
    else :
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20,"B":21,"Z":22,"X":23}
    if zero_characters:
        for element in zero_characters:
                aminoacid_names[element] = 0
    aminoacid_names = {k: v for k, v in sorted(aminoacid_names.items(), key=lambda item: item[1])} #sort dict by values (for dicts it is an overkill, but I like ordered stuff)
    return aminoacid_names
def score_match(pair, matrix):
    """Returns the corresponding blosum scores between the pair of amino acids
    :param tuple pair: pair of amino acids to compare
    :param matrix: Blosum matrix """
    if pair not in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return matrix[pair]
def score_pairwise(seq1, seq2, matrix, gap_s, gap_e): #TODO: remove
    """
    Calculates the blosum score of the true sequence against the predictions
    :param matrix matrix: Blosum matrix containing the log-odds scores (he logarithm for the ratio of the
       likelihood of two amino acids appearing with a biological sense and the likelihood of the same amino acids appearing by chance)
        (the higher the score, the more likely the corresponding amino-acid substitution is)
    :param int gap_s: gap penalty, 11
    :param int gap_e : mismatch penalty gap_s=11, 1
    """
    score = 0
    gap = False
    for i in range(len(seq1)):
        pair = (seq1[i], seq2[i])
        if not gap:
            if '-' in pair:
                gap = True
                score += gap_s
            # elif "*" in pair: #TODO: Keep?
            #     score +=gap_e
            else:
                score += score_match(pair, matrix)
        else:
            if '-' not in pair:# and "*" not in pair:
                gap = False
                score += score_match(pair, matrix)
            else:
                gap = True
                score += gap_e
    return score



def create_blosum(aa_types,subs_matrix_name):
    """
    Builds an array containing the blosum scores per character
    :param aa_types: amino acid probabilities, determines the choice of BLOSUM matrix
    :param str subs_matrix_name: name of the substitution matrix, check availability at /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data"""

    if aa_types > 20 and not subs_matrix_name.startswith("PAM"):
        warnings.warn("Your dataset contains special amino acids. Switching your substitution matrix to PAM70")
        subs_matrix_name = "PAM70"
    subs_matrix = Bio.Align.substitution_matrices.load(subs_matrix_name)
    aa_list = list(aminoacid_names_dict(aa_types,zero_characters=["#"]).keys())
    index_gap = aa_list.index("#")
    aa_list[index_gap] = "*" #in the blosum matrix gaps are represanted as *

    subs_dict = defaultdict()
    subs_array = np.zeros((len(aa_list) , len(aa_list) ))
    for i, aa_1 in enumerate(aa_list):
        for j, aa_2 in enumerate(aa_list):
            if aa_1 != "*" and aa_2 != "*":
                subs_dict[(aa_1,aa_2)] = subs_matrix[(aa_1, aa_2)]
                subs_dict[(aa_2, aa_1)] = subs_matrix[(aa_1, aa_2)]
            else:
                subs_dict[(aa_1, aa_2)] = -1 #gap penalty

            subs_array[i, j] = subs_matrix[(aa_1, aa_2)]
            subs_array[j, i] = subs_matrix[(aa_2, aa_1)]

    names = np.concatenate((np.array([float("-inf")]), np.arange(0,len(aa_list))))
    subs_array = np.c_[ np.arange(0,len(aa_list)), subs_array ]
    subs_array = np.concatenate((names[None,:],subs_array),axis=0)
    #subs_array[1] = np.zeros(aa_types+1)  #replace the gap scores for zeroes , instead of [-4,-4,-4...]
    #subs_array[:,1] = np.zeros(aa_types+1)  #replace the gap scores for zeroes , instead of [-4,-4,-4...]

    #blosum_array_dict = dict(enumerate(subs_array[1:,2:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements
    blosum_array_dict = dict(enumerate(subs_array[1:,1:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements
    print(blosum_array_dict)
    exit()

    return subs_array, subs_dict, blosum_array_dict