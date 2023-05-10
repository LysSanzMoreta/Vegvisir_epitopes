import os
import matplotlib.pyplot as plt
import numpy as np
from prody import *
from Bio import SeqIO,SeqRecord
from Bio.Seq import Seq
def create_fasta_file(seq_list,ids,file_name,results_dir,dataset_name):
    print("Fasta file does not exist, creating it")
    record_list = []
    for seq,id in zip(seq_list,ids):
        record = SeqRecord.SeqRecord(Seq("".join(seq).replace("#","-")),
                                     annotations={"molecule_type": "protein"}, id=str(id), description="")
        record_list.append(record)
    SeqIO.write(record_list,"{}/{}/{}.fasta".format(results_dir,dataset_name,file_name),"fasta")


def calculate_mutual_information(seq_list,ids,max_len,mode,results_dir,dataset_name):

    if not os.path.exists("{}/{}/{}.fasta".format(results_dir,dataset_name,mode)):
        create_fasta_file(seq_list,ids,mode,results_dir,dataset_name)
    msa = parseMSA("{}/{}/{}.fasta".format(results_dir,dataset_name,mode))
    msa_mi =  buildDirectInfoMatrix(msa)

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,6))

    im = ax[0].imshow(msa_mi)
    ax[0].set_xticks(np.arange(max_len),labels=list(range(max_len)))
    ax[1].axis("off")
    cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cb_ax)
    fig.suptitle("Mutual information {}".format(mode))
    plt.savefig("{}/{}/Mutual_Information{}".format(results_dir,dataset_name,mode))

def calculate_fisher_information():
    """
    Notes:
    https://timeseriesreasoning.com/contents/fisher-information/
    https://wittman.physics.ucdavis.edu/Fisher-matrix-guide.pdf
    https://stackoverflow.com/questions/48695308/fisher-information-calculation-extended
    """

