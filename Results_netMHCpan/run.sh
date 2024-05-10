#!/bin/bash


alleles="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/alleles_list.txt"


foldertype="Generated"
samplingtype="Independent_sampling"
#foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_27_19h15min48s745646ms_60epochs_supervised_Icore_blosum_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype
##python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/${foldername}_${foldertype} -folder-type Generated
#
#foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_27_19h30min55s270247ms_60epochs_supervised_Icore_blosum_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype
#
#foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_27_19h46min03s397524ms_60epochs_supervised_Icore_blosum_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername=""
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_28_15h08min36s122719ms_100epochs_supervised_Icore_blosum"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_28_15h23min04s900117ms_100epochs_supervised_Icore_blosum"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_28_15h37min25s194518ms_100epochs_supervised_Icore_blosum"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_28_15h51min54s701614ms_100epochs_supervised_Icore_blosum"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_28_16h06min07s433770ms_100epochs_supervised_Icore_blosum"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_28_16h20min27s835979ms_100epochs_supervised_Icore_blosum"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

###############################################################################################################################

foldertype="Generated"
samplingtype="Conditional_sampling"
#foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_27_19h25min02s448145ms_60epochs_supervised_Icore_blosum_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype
##python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/${foldername}_${foldertype} -folder-type Generated
#
#foldername="PLOTS_Vegvisir_viral_dataset15_2024_03_27_19h40min20s623600ms_60epochs_supervised_Icore_blosum_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

#foldername=""
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype

#foldername=""
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/$foldertype/epitopes.txt -ft $foldertype