import sys
import os
local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__))
if local_repository: #TODO: The local imports are extremely slow
     sys.path.insert(1, "{}/vegvisir/src".format(script_dir))
     import vegvisir
else:#pip installed module
     import vegvisir
import vegvisir.plots as VegvisirPlots

def analysis_models(args):
    """Analyses the results of all possible model combinations (stress testing)"""

    #Highlight: Supervised
    dict_results_predefined_partitions_viral_dataset15_HPO_z16 = {"Icore": {
        "random-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_00h04min32s272010ms_60epochs_supervised_Icore_blosum_random_TESTING",
        "random-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_01h09min17s856077ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
        "shuffled-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_01h52min54s450522ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
        "shuffled-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_02h55min09s838797ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
        "shuffled-labels-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_03h39min37s644453ms_60epochs_supervised_Icore_blosum_shuffled_labels_TESTING",
        "shuffled-labels-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_04h42min48s726967ms_60epochs_supervised_Icore_blosum_shuffled_labels_TESTING_9mers",
        "raw-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_05h26min27s211598ms_60epochs_supervised_Icore_blosum_TESTING",
        "raw-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_06h30min09s527987ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
    },
        "Icore_non_anchor": {
            "random-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_00h05min18s810754ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
            "random-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_01h09min17s668244ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_7mers",
            "shuffled-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_01h40min30s015215ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
            "shuffled-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_02h42min18s960420ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_7mers",
            "shuffled-labels-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_03h12min55s905949ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_labels_TESTING",
            "shuffled-labels-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_04h14min16s489968ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_labels_TESTING_7mers",
            "raw-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_04h44min53s870813ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
            "raw-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_05h46min44s903138ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_7mers"
        }}


    #Highlight: Semi-supervised

    dict_results_predefined_partitions_viral_dataset17 = {"Icore": {
        "random-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_10h39min03s813549ms_60epochs_semisupervised_Icore_blosum_random_TESTING_5000_unobserved",
        "random-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_13h23min16s713572ms_60epochs_semisupervised_Icore_blosum_random_TESTING_9mers_5000_unobserved",
        "shuffled-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_14h54min07s936592ms_60epochs_semisupervised_Icore_blosum_shuffled_TESTING_5000_unobserved",
        "shuffled-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_17h36min19s432574ms_60epochs_semisupervised_Icore_blosum_shuffled_TESTING_9mers_5000_unobserved",
        "shuffled-labels-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_19h07min41s320049ms_60epochs_semisupervised_Icore_blosum_shuffled_labels_TESTING_5000_unobserved",
        "shuffled-labels-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_21h48min47s818295ms_60epochs_semisupervised_Icore_blosum_shuffled_labels_TESTING_9mers_5000_unobserved",
        "raw-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_21_23h20min35s932476ms_60epochs_semisupervised_Icore_blosum_TESTING_5000_unobserved",
        "raw-9mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_22_02h02min28s362838ms_60epochs_semisupervised_Icore_blosum_TESTING_9mers_5000_unobserved",
    },
        "Icore_non_anchor": {
            "random-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_22_18h25min07s708296ms_60epochs_semisupervised_Icore_non_anchor_blosum_random_TESTING_5000_unobserved",
            "random-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_22_21h07min26s251718ms_60epochs_semisupervised_Icore_non_anchor_blosum_random_TESTING_7mers_5000_unobserved",
            "shuffled-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_22_22h13min05s677394ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_TESTING_5000_unobserved",
            "shuffled-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_23_00h55min57s186121ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_TESTING_7mers_5000_unobserved",
            "shuffled-labels-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_23_02h01min14s314893ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_labels_TESTING_5000_unobserved",
            "shuffled-labels-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_23_04h44min02s238828ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_labels_TESTING_7mers_5000_unobserved",
            "raw-variable-length": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_23_21h42min23s591302ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_5000_unobserved",
            "raw-7mers": "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset17/PLOTS_Vegvisir_viral_dataset17_2024_10_24_00h18min58s121246ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_7mers_5000_unobserved"
        }}

    #Highlight: K-fold comparisons

    # VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_viral_dataset15_HPO_z16,kfolds=5,results_folder = "Benchmark/Plots",title="VIRAL_DATASET15_HPO",overwrite=False)
    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_viral_dataset17,kfolds=5,results_folder = "Benchmark/Plots",title="VIRAL_DATASET17_HPO",overwrite=False)


    #VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_predefined_partitions_viral_dataset15_HPO_z16,kfolds=5,results_folder="Benchmark/Plots",subtitle="VIRAL_DATASET15_HPO_z16_tryptophan",overwrite_correlations=False,overwrite_all=False)
    #VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_predefined_partitions_viral_dataset17,kfolds=5,results_folder="Benchmark/Plots",subtitle="VIRAL_DATASET17_HPO",overwrite_correlations=False,overwrite_all=False)


    dict_results_benchmark= { "Icore" :{
        "raw-variable-length-vd15":dict_results_predefined_partitions_viral_dataset15_HPO_z16["Icore"]["raw-variable-length"],
        "raw-variable-length-vd17":dict_results_predefined_partitions_viral_dataset17["Icore"]["raw-variable-length"],
    }}

    results_key = "raw-variable-length-vd15"
    #results_key = "raw-variable-length-vd17"
    title = "VIRAL_DATASET15_HPO_z16"
    #title = "VIRAL_DATASET17_HPO"
    #Highlight: Benchmarking
    #Highlight: Supervised
    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,keyname=results_key,
                                            folder="Benchmark/Plots",
                                            title=f"{title}_only_overlapped_seqs_ENSEMBL",
                                            keep_only_overlapped=True,aggregated_not_overlap=False,keep_all=False,only_class1=False,ensemble=True)


    # ##Highlight: Supervised
    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,keyname=results_key,
                                            folder="Benchmark/Plots",title=f"{title}_removed_ALL_overlapping_seqs_ENSEMBL",
                                            keep_only_overlapped=False,aggregated_not_overlap=True,keep_all=False,only_class1=False,ensemble=True)


    # ##Highlight: Supervised
    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,keyname=results_key,
                                            folder="Benchmark/Plots",title=f"{title}_ENSEMBL",
                                            keep_only_overlapped=False,aggregated_not_overlap=False,keep_all=True,only_class1=False,ensemble=True) #keeps all sequences


    #Highlight: Supervised
    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,keyname=results_key,
                                            folder="Benchmark/Plots",title=f"{title}_Vegvisir_keep_all_common_sequences_ENSEMBL_NEWWWWW",
                                            keep_only_overlapped=True,aggregated_not_overlap=True,only_class1=True,ensemble=True)


    # ##Highlight: Supervised
    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,keyname=results_key,folder="Benchmark/Plots",
                                            title=f"{title}_Vegvisir_removed_overlapping_sequences_per_model_ENSEMBL",
                                            keep_only_overlapped=False,aggregated_not_overlap=False,keep_all=False,only_class1=False,ensemble=True)



    #Highlight: Supervised: Model stress comparison
    #VegvisirPlots.plot_model_stressing_comparison1(dict_results_predefined_partitions_viral_dataset9_HPO_z4,script_dir,results_folder="Benchmark/Plots",encoding="-",subtitle="VIRAL_DATASET9_HPO_z4",keyname="viral_dataset9")

    #
    # VegvisirPlots.plot_model_stressing_comparison2(dict_results_predefined_partitions_viral_dataset15_HPO_z16,script_dir,results_folder="Benchmark/Plots",
    #                                                encoding="-",subtitle=f"{title}_ONLY_VEGVISIR_ENSEMBL",keyname="viral_dataset15",ensemble=True)

    # Highlight: Semisupervised: Model stress comparison

    VegvisirPlots.plot_model_stressing_comparison2(dict_results_predefined_partitions_viral_dataset17,script_dir,results_folder="Benchmark/Plots",
                                                   encoding="-",subtitle="VIRAL_DATASET17_ONLY_VEGVISIR_ENSEMBL",keyname="viral_dataset17",ensemble=True)



def hierarchical_clustering():

    vegvisir_folder_z34 = "PLOTS_Vegvisir_viral_dataset9_2023_12_26_18h37min00s675744ms_60epochs_supervised_Icore_blosum_TESTING_z34"
    vegvisir_folder_viral_dataset_9_z4 = "Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_23h31min43s312787ms_60epochs_supervised_Icore_blosum_TESTING"
    vegvisir_folder_viral_dataset_9_z30 = "PLOTS_Vegvisir_viral_dataset13_2024_01_05_21h14min29s243245ms_60epochs_supervised_Icore_60_TESTING_z30"
    vegvisir_viral_dataset15_z16 = "Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_05h26min27s211598ms_60epochs_supervised_Icore_blosum_TESTING"
    external_paths_dict = {"embedded_epitopes":"{}/vegvisir/src/vegvisir/data/viral_dataset15/similarities/Icore/All/diff_allele/diff_len/neighbours1/all/EMBEDDED_epitopes.tsv".format(script_dir),
                           #"esmb1_path":"{}/vegvisir/src/vegvisir/data/common_files/Epitopes_info_TRAIN_esmb1.tsv".format(script_dir),
                           "esmb1_path":"{}/vegvisir/src/vegvisir/data/common_files/ESMB1_all.csv".format(script_dir)
                           }
    #VegvisirPlots.plot_hierarchical_clustering(vegvisir_folder_viral_dataset_9_z4, external_paths_dict,folder="Benchmark/Plots",title="VIRAL_DATASET9_HPO_z9_tryptophan",keyname="viral_dataset9")
    VegvisirPlots.plot_hierarchical_clustering(vegvisir_viral_dataset15_z16, external_paths_dict,folder="Benchmark/Plots",title="VIRAL_DATASET15_HPO_z16_tryptophan",keyname="viral_dataset15")

def ablation_study():
    #Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_25_22h25min27s788645ms_60epochs_supervised_Icore_blosum_TESTING
    ablation_dict = { 1:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_25_22h25min27s788645ms_60epochs_supervised_Icore_blosum_TESTING",
                     10:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_25_22h55min51s373033ms_60epochs_supervised_Icore_blosum_TESTING",
                     20:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_25_23h25min33s535839ms_60epochs_supervised_Icore_blosum_TESTING",
                     30:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_25_23h56min02s283158ms_60epochs_supervised_Icore_blosum_TESTING",
                     40:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_26_00h25min27s753388ms_60epochs_supervised_Icore_blosum_TESTING",
                     50:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_26_00h55min37s940818ms_60epochs_supervised_Icore_blosum_TESTING",
                     60:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_26_01h27min12s342080ms_60epochs_supervised_Icore_blosum_TESTING",
                     70:"Ablation_studies/Likelihood/Ablation_study_likelihood/PLOTS_Vegvisir_viral_dataset15_2024_03_26_02h01min50s511689ms_60epochs_supervised_Icore_blosum_TESTING",
                     }
    VegvisirPlots.plot_ablation_study(ablation_dict,script_dir,"Ablation_studies/Likelihood",subtitle="",ensemble=True)


    vegvisir_dict = {"raw-variable-length":"Benchmark/Vegvisir_benchmarking/HPO_viral_dataset15/PLOTS_Vegvisir_viral_dataset15_2024_01_19_05h26min27s211598ms_60epochs_supervised_Icore_blosum_TESTING"}

    #VegvisirPlots.plot_metrics_per_length(vegvisir_dict,script_dir,folder="Ablation_studies/Lengths",subtitle="")


