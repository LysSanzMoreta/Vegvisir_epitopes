
#!/bin/bash



alleles="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/alleles_list.txt"

#foldername="PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING"
foldername="PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING_generated"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Immunomodulated/generated_epitopes.txt
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/generated_epitopes.txt
#python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername