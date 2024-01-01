
#!/bin/bash



alleles="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/alleles_list.txt"

#foldername="PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING_generated"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_21_09h48min25s760830ms_60epochs_supervised_Icore_blosum_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_21_18h17min01s865709ms_60epochs_supervised_Icore_blosum_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_26_21h41min44s738321ms_0epochs_supervised_Icore_0_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_27_16h18min53s821190ms_0epochs_supervised_Icore_0_TESTING"
#foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_29_15h31min21s577193ms_0epochs_supervised_Icore_0_TESTING"
#bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Immunomodulated/generated_epitopes.txt
#python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Immunomodulated
foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_29_19h32min20s745052ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated

foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_29_21h02min38s346973ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated

foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_29_22h30min58s427689ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated

foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_30_12h26min26s976339ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated

foldername="PLOTS_Vegvisir_viral_dataset9_2023_12_30_13h43min42s487903ms_60epochs_supervised_Icore_blosum_TESTING"
bash run_netmhcpan.sh -a $alleles -p /home/lys/Dropbox/PostDoc/vegvisir/$foldername/Generated/epitopes.txt
python process_results.py -folder-path /home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/$foldername -folder-name Generated