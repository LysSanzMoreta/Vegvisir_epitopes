#REMEMBER TO CALL USING :
# bash run_netmhcpan.sh -a /home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/alleles_list.txt -p /home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_09_12_15h33min42s969996ms_60epochs_supervised_Icore_onehot_TESTING/Generated/generated_epitopes.txt
#!/bin/bash

#First detect arguments
#alleles
if getopts "a:" arg; then
  echo "Allele path: $OPTARG"
    alleles=$OPTARG
fi
echo $arg
#peptides
if getopts "p:" arg; then
  echo "Generated peptides path:  $OPTARG"
    peptides=$OPTARG
fi
echo $arg
#results path #TODO: This is not working

if getopts "r:" arg; then
      echo "Results path:  $OPTARG"
      results=$OPTARG
      echo "Here1"
else
      results="/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan"
      echo "Results path : $results"
fi
results="/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan"


echo $arg
#folder type
if getopts "ft:" arg; then
      echo "Folder type:  $OPTARG"
      foldertype=$OPTARG
      echo "Here2"
else
      foldertype="Generated"
      echo "Folder type : $foldertype"
fi
echo $arg
############################################

shopt -s expand_aliases
alias netmhcpan='sh /home/lys/netMHCpan-4.1/netMHCpan'

##Create directory with current date/time to store the results
#time_stamp=$(date +%Y_%m_%d_%T)
#echo $time_stamp
foldername=$(basename $(dirname $(dirname "$peptides")))
mkdir -p "${results}/${foldername}_${foldertype}" #does not create a new folder if exists
echo "${results}/${foldername}_${foldertype}"


#While loop to assess alleles
counter=0
cat $alleles | while read line
echo "Starting while loop"
do
  echo $line
  if [[ "$line" == "" ]]; then
    echo "Reached empty line"
    break
  fi
  echo "Running NetMHCpan with alleles: $line"
  echo "$results/${foldername}_${foldertype}/group${counter}_alleles.txt"
  netmhcpan -p $peptides -a $line -l , > "$results/${foldername}_${foldertype}/group${counter}_alleles"
  counter=$((counter+=1))
done