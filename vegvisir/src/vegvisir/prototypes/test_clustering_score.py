import numpy as np
from collections import defaultdict

from scipy.stats import permutation_test,mannwhitneyu,rankdata
#https://leetcode.com/problems/max-consecutive-ones/solutions/4366559/python-easy-98-beats/
#find starting index
def clustering_accuracy(binary_arr):
    """Computes a clustering accuracy score based on the maximum of size of a cluster (and spreadness)"""
    maxlen = binary_arr.shape[0]
    count_ones, count_zeros = 0, 0
    max_count_ones, max_count_zeros = 0, 0
    previdx_ones, previdx_zeros = 0, 0
    groups_counts_ones, groups_counts_zeros = defaultdict(), defaultdict() #registers the start index of the 1ś clusters
    for idx,num in enumerate(binary_arr):
        if num != 1:
            max_count_ones = max(max_count_ones, count_ones)
            if idx == 0:
                groups_counts_ones[previdx_ones] = count_ones #previous index plus 1
            else:
                groups_counts_ones[previdx_ones +1] = count_ones #previous index plus 1
            previdx_ones=idx
            count_ones = 0
            count_zeros += 1
        else:
            max_count_zeros = max(max_count_zeros, count_zeros)
            if idx == 0:
                groups_counts_zeros[previdx_zeros] = count_zeros #previous index plus 1
            else:
                groups_counts_zeros[previdx_zeros + 1] = count_zeros #previous index plus 1
            count_zeros = 0
            previdx_zeros=idx
            count_ones += 1

    if previdx_zeros + 1 < maxlen:
        groups_counts_zeros[previdx_zeros + 1] = count_zeros #previous index plus 1, have to add this here
    if previdx_ones + 1 < maxlen:
        groups_counts_ones[previdx_ones + 1] = count_ones #previous index plus 1, have to add this here

    maxcountones = max(max_count_ones, count_ones)
    maxcountzeros = max(max_count_zeros, count_zeros)
    total_std_idx = np.std(np.arange(maxlen))
    total_ones = np.sum(binary_arr)
    total_zeros = binary_arr.shape[0] - total_ones


    # Highlight: Transform to array keeping only the results for the ones
    starting_idx_ones = np.array([key for key, val in groups_counts_ones.items() if val != 0])
    idx_ones = np.array(binary_arr == 1)
    idx_ones = np.arange(maxlen)[idx_ones]
    cluster_sizes_ones = np.array([val for key, val in groups_counts_ones.items() if val != 0])
    #counts_array_ones = np.concatenate([starting_idx[:, None], cluster_size[:, None]], axis=1)

    if cluster_sizes_ones.size != 0:
        std_idx_ones = np.std(idx_ones)
        max_size_ones = np.max(cluster_sizes_ones)
    else:
        std_idx_ones = total_std_idx
        max_size_ones = total_ones = 1

    starting_idx_zeros = np.array([key for key, val in groups_counts_zeros.items() if val != 0])
    idx_zeros = np.array(binary_arr == 0)
    idx_zeros = np.arange(maxlen)[idx_zeros]
    cluster_sizes_zeros = np.array([val for key, val in groups_counts_zeros.items() if val != 0])

    if cluster_sizes_zeros.size != 0:
        std_idx_zeros = np.std(idx_zeros)
        max_size_zeros = np.max(cluster_sizes_zeros)
    else:
        std_idx_zeros = total_std_idx
        max_size_zeros = total_zeros = 1

    score_a = ((max_size_ones*100/total_ones) + (max_size_zeros*100/total_zeros)) / 2
    score_b = ((std_idx_ones*100/total_std_idx) + (std_idx_zeros*100/total_std_idx) + 2*(max_size_ones*100/total_ones) + 2*(max_size_zeros*100/total_zeros)) / 6

    return {"maxcountones": maxcountones,
            "group_counts_ones": groups_counts_ones,
            "maxcountzeros": maxcountzeros,
            "group_counts_zeros": groups_counts_zeros,
            "clustering_score_a" : score_a,
            "clustering_score_b" : score_b
            }



def clustering_significance(labels):
    """Performs a permutation test on the location of the labels (idx label 0 or idx label 1) to estimate the significance of the clusters, whether they are due to random or not ...
    NOTES:
        For a given array of clustered labels of size N, calculate the average rank of one of the two labels, say label 1. Call this value <R_1>

        Next repeat 10000 times:permute the order of the array of clustered labels (each time with a new seed)
                calculate the average rank of the entries with label 1, <R_1_perm>
        Next,
        if <R_1>  < N/2:
                the p-value for the rank of label 1 entries in the original data is random will be equal to the proportion of <R_1_perm> values that are lower than <R_1>
        else
                the p-value for the rank of label 1 entries in the original data is random will be equal to the proportion of <R_1_perm> values that are higher than <R_1>
    """

    idx_ones = np.where(labels==1)[0].astype(float)
    ndata = labels.shape[0]
    r1_avg = np.average(idx_ones)
    r1_permuted_list = []
    for i in range(10): #test loop
        np.random.seed(i)
        shuffled_labels = labels.copy()
        np.random.shuffle(shuffled_labels)
        idx_ones_shuffled = np.where(shuffled_labels==1)[0].astype(float)
        r1_permuted = np.average(idx_ones_shuffled)
        r1_permuted_list.append(r1_permuted)
    r1_permuted_arr = np.array(r1_permuted_list)
    if r1_avg < ndata/2:
        lower_idx = np.where(r1_permuted_arr < r1_avg)[0]

        pval = len(lower_idx)/ndata
    else:
        higher_idx = np.where(r1_permuted_arr > r1_avg)[0]
        pval = len(higher_idx)/ndata
    return pval


a = np.array([0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,0])
b = np.array([0,1,1,1,1,0,0,1,1,0,0,0,0,0,1,0])
c = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
d = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
e = np.array([0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0])
f = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])


clustering_significance(a)
clustering_significance(b)
clustering_significance(c)
clustering_significance(d)
clustering_significance(e)
clustering_significance(f)

exit()


cluster_analysis_dict = clustering_accuracy(a)
print(cluster_analysis_dict["clustering_score_a"])

cluster_analysis_dict = clustering_accuracy(b)
print(cluster_analysis_dict["clustering_score_a"])

cluster_analysis_dict = clustering_accuracy(c)
print(cluster_analysis_dict["clustering_score_a"])

cluster_analysis_dict = clustering_accuracy(d)
print(cluster_analysis_dict["clustering_score_a"])

cluster_analysis_dict = clustering_accuracy(e)
print(cluster_analysis_dict["clustering_score_a"])

cluster_analysis_dict = clustering_accuracy(f)
print(cluster_analysis_dict["clustering_score_a"])

