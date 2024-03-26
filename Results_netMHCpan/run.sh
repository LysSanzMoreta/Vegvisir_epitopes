#!/bin/bash



alleles="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/alleles_list.txt"


foldertype="Generated"
samplingtype="Conditional_sampling"
foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_26_22h14min38s909541ms_2epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype
#python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/${foldername}_${foldertype} -folder-type Generated

