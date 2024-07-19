#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from prody import *
from Bio import SeqIO,SeqRecord
from Bio.Seq import Seq
from sklearn.metrics import mutual_info_score


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


def calculate_mi(data,data_mask,aa_groups,max_len,mode,results_dir,dataset_name,analysis_mode,save_plot=True):
    """
    I(X,Y) = H(X,Y) - H(Y|X) - H(X|Y) where I is the Information theory and H is the entropy
    :param data:
    :param data_mask:
    :param aa_groups:
    :return:
    Notes:
        -https://artem.sobolev.name/posts/2019-09-15-thoughts-on-mutual-information-alternative-dependency-measures.html
    """
    print("Calculating Mutual information")

    if data.size != 0:
        n_data = data.shape[0]
        data_idx = list(range(max_len))
        mi_matrix = np.zeros((max_len,max_len))

        for i in data_idx: #for site in the sequence
            if i+1 <= max_len:
                for j in data_idx[i+1:]: #for next site in the sequence

                    #r = computeMI(data[:,i],data[:,j])
                    mi = mutual_info_score(data[:,i],data[:,j])

                    mi_matrix[i, j] = mi


        if save_plot:
            fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,6))
            max_mi = np.argmax(mi_matrix)
            im = ax[0].imshow(mi_matrix) #vmin=0, vmax=1.5
            ax[0].set_xticks(np.arange(max_len),labels=list(range(max_len)))
            ax[1].axis("off")
            cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
            fig.colorbar(im, cax=cb_ax)
            fig.suptitle("Mutual information {}".format(mode))
            if analysis_mode and mode:
                plt.savefig("{}/{}/{}/Mutual_Information{}".format(results_dir,dataset_name,analysis_mode,mode),dpi=600)
            else:
                plt.savefig("{}/Mutual_Information".format(results_dir),dpi=600)
            plt.close(fig)
        return mi_matrix
    else:
        print("Empty dataset, not calculating mutual information")




