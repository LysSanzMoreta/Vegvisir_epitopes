import os
import matplotlib.pyplot as plt
import numpy as np
from prody import *
from Bio import SeqIO,SeqRecord
from Bio.Seq import Seq
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



if __name__ == '__main__':

    max_len = 4
    seqs = ["AHPD","ALSW","VLPY","TRMF","IKNM"]#,"FYRA"]
    ids = ["0","1","2","3","4"]

    calculate_mutual_information(seqs, ids, max_len, "test_MI", "")