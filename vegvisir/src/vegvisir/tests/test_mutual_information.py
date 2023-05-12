import os,sys
import matplotlib.pyplot as plt
import numpy as np
from prody import *
from Bio import SeqIO,SeqRecord
from Bio.Seq import Seq
local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__))
if local_repository: #TODO: The local imports are extremely slow
     sys.path.insert(1, "/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src")
     import vegvisir
else:#pip installed module
     import vegvisir
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.utils as VegvisirUtils
def create_fasta_file(seq_list,ids,file_name,results_dir):
    print("Fasta file does not exist, creating it")
    record_list = []
    for seq,id in zip(seq_list,ids):
        record = SeqRecord.SeqRecord(Seq(seq),
                                     annotations={"molecule_type": "protein"}, id=id, description="")
        record_list.append(record)
    SeqIO.write(record_list,"{}.fasta".format(file_name),"fasta")


def calculate_mutual_information(seq_list,ids,max_len,file_name,results_dir):

    if not os.path.exists("{}.fasta".format(file_name)):

        create_fasta_file(seq_list,ids,file_name,results_dir)
    msa = parseMSA("{}.fasta".format(file_name))
    msa_mi =  buildDirectInfoMatrix(msa)

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,6))

    im = ax[0].imshow(msa_mi)
    ax[0].set_xticks(np.arange(max_len),labels=list(range(max_len)))
    ax[1].axis("off")
    cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cb_ax)
    plt.show()

def calculate_aa_frequencies(dataset,freq_bins):
    """Calculates a frequency for each of the aa & gap at each position.The number of bins (of size 1) is one larger than the largest value in x. This is done for numpy arrays
    :param tensor dataset
    :param int freq_bins
    """

    freqs = np.apply_along_axis(lambda x: np.bincount(x, minlength=freq_bins), axis=0, arr=dataset.astype("int64")).T
    #freqs = freqs/dataset.shape[0]
    return freqs
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = np.nansum(mutual.as_nd_ndarray())
    return out


def computeMI(x, y):
    """https://stackoverflow.com/questions/24686374/pythons-implementation-of-mutual-information"""
    print("x :{}".format(x))
    print("y :{}".format(y))

    sum_mi = 0.0
    x_unique = np.unique(x) #unique groups of amino acids in this site
    print("x unique: {}".format(x_unique))
    y_unique = np.unique(y) #unique groups of amino acids in the other site
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_unique ]) #P(x) or frequency/n_data for each of the present groups
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_unique ]) #P(y)

    for i in range(len(x_unique)): #for each unique value in the site i
        print("x_unique[i] {}".format(x_unique[i]))
        if Px[i] ==0.: #hmmm, not sure with the previous code when would the probability be 0
            continue
        print("####################33")
        print(x == x_unique[i])
        print("####################33")
        sy = y[x == x_unique[i]]
        print("sy {}".format(sy))
        if len(sy)== 0: #if there are no values pass ..
            continue
        #for each unique value in site 2 , if that value is found in the same position
        print("y unique {}".format(y_unique))

        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_unique]) #p(x,y)

        t = (pxy[Py>0.]/Py[Py>0.]) /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
        print("sum mi")
        print(sum_mi)
    return sum_mi
def calculate_mi(data,data_mask,aa_groups):
    """
    I(X,Y) = H(X,Y) - H(Y|X) - H(X|Y) where I is the Information theory and H is the entropy
    :param data:
    :param data_mask:
    :param aa_groups:
    :return:
    """
    from sklearn.metrics import mutual_info_score
    n_data = data.shape[0]
    frequencies = calculate_aa_frequencies(data,aa_groups) #[L,aa_groups]
    print("data set")
    print(data)
    probabilities = frequencies/n_data
    data_idx = list(range(max_len))
    mi_matrix = np.zeros((max_len,max_len))
    mi_matrix_2 = np.zeros((max_len,max_len))

    for i in data_idx: #for site in the sequence
        if i+1 < max_len:
            for j in data_idx[i+1:]: #for next site in the sequence
                #calculate the frequency of each aa group
                counts_site_i = frequencies[i]
                counts_site_j = frequencies[j]
                #r = computeMI(data[:,i],data[:,j])
                r2 = mutual_info_score(data[:,i],data[:,j])

                #mi_matrix[i,j] = r
                mi_matrix_2[i, j] = r2
                #nonzeros_i = np.nonzero(data_idx[i])
                #nonzeros_j = np.nonzero(data_idx[j])

    print(mi_matrix)
    print(mi_matrix_2)

    return mi_matrix_2















if __name__ == '__main__':
    """
    Notes: 
        https://github.com/babylonhealth/corrsim
        https://www.blog.trainindata.com/mutual-information-with-python/
        https://timeseriesreasoning.com/contents/fisher-information/
        https://wittman.physics.ucdavis.edu/Fisher-matrix-guide.pdf
        https://stackoverflow.com/questions/48695308/fisher-information-calculation-extended
        http://www.sefidian.com/2017/06/14/mutual-information-mi-and-entropy-implementations-in-python/
        CODE: https://gist.github.com/TheLoneNut/208cd69bbca7cd7c53af26470581ec1e
        https://www.kaggle.com/code/ryanholbrook/mutual-information
        CYTHON CODE: https://www.pik-potsdam.de/~donges/pyunicorn/_modules/pyunicorn/climate/mutual_info.html
    """

    max_len = 4
    seqs = ["AHPD",
            "ALSW",
            "VLPY",
            "TRMF",
            "IKNM"]#,"FYRA"]
    ids = ["0","1","2","3","4"]
    aa_types = 21
    sequences_padded = VegvisirLoadUtils.SequencePadding(seqs, max_len, "ends",False).run()
    sequences, mask  = zip(*sequences_padded)  # unpack list of tuples onto 2 lists
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(aa_types, "BLOSUM62",
                                                                               zero_characters=["#"],
                                                                               include_zero_characters=True)
    sequences_array = np.array(sequences)

    aa_dict = VegvisirUtils.aminoacid_names_dict(aa_types, zero_characters=["#"])
    sequences_int = np.vectorize(aa_dict.get)(sequences_array)
    sequences_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(sequences_int)
    aa_groups_colors_dict, aa_groups_dict, groups_names_colors_dict = VegvisirUtils.aminoacids_groups(aa_dict) #TODO: Gropus by blosum cosine similarity (group aas with similar blosum vectors)
    aa_groups = len(groups_names_colors_dict.keys())
    sequences_int_group = np.vectorize(aa_groups_dict.get)(sequences_int)
    #sequences_mask = np.ones_like(sequences_blosum).astype(bool)[:,:,0]
    sequences_mask = np.array([[True,True,True,False],[True,True,True,True],[True,False,False,False],[True,True,True,False],[True,True,False,False]])#,[True,True,True,True]])

    #calculate_mutual_information(seqs, ids, max_len, "test_MI", "")

    calculate_mi(sequences_int_group,sequences_mask,aa_groups)