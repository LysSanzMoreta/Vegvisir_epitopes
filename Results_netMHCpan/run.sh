
#!/bin/bash



alleles="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/alleles_list.txt"

#foldername="PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING_generated"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_21_09h48min25s760830ms_60epochs_supervised_Icore_blosum_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_21_18h17min01s865709ms_60epochs_supervised_Icore_blosum_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_26_21h41min44s738321ms_0epochs_supervised_Icore_0_TESTING" #BEST ONE
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_27_16h18min53s821190ms_0epochs_supervised_Icore_0_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_29_15h31min21s577193ms_0epochs_supervised_Icore_0_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Immunomodulated/generated_epitopes.txt
#python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Immunomodulated
##############################################################################################################
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/generated_epitopes.txt
#python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated
#exit 0


#
foldername="PLOTS_Vegvisir_viral_dataset9_2024_01_03_10h36min19s973660ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated
#
foldername="PLOTS_Vegvisir_viral_dataset9_2024_01_03_11h07min13s245758ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated

foldername="PLOTS_Vegvisir_viral_dataset9_2024_01_03_11h36min29s509173ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated
#
foldername="PLOTS_Vegvisir_viral_dataset9_2024_01_03_12h43min32s587030ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated

foldername="PLOTS_Vegvisir_viral_dataset9_2024_01_03_13h59min39s274151ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated