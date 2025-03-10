#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""
import json
import logging
import multiprocessing
import os
import shutil
from argparse import Namespace
from operator import is_not

import numpy as np
import vegvisir
import vegvisir.train_svi as VegvisirTrainSVI
from collections import namedtuple
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial

AdditionalInfo = namedtuple("AdditionalInfo",["results_dir"])

def hyperparameter_optimization(dataset_info:namedtuple,additional_info:namedtuple,args:namedtuple):
    """Initiates Hyperparameter search with Ray-tune
    Notes:
        -Adam parameters: https://www.kdnuggets.com/2022/12/tuning-adam-optimizer-parameters-pytorch.html
    """

    print("Initiating Hyperparameter Optimization ....")

    max_num_epochs = 60

    config1 = {
        "lr":tune.choice([1e-4,1e-3,1e-2]),
        "beta1": tune.choice([0.9,0.91,0.92,0.93,0.94]),
        "beta2": tune.choice([0.999,0.9999]),
        "eps": tune.choice([1e-8]), #not tuning parameter, avoids division by 0
        "weight_decay": tune.choice([0]),
        "clip_norm":tune.choice([10]), #not used with pyro
        "lrd":tune.choice([1]),
        "momentum":tune.randn(0.9,0.1)
    }
    config2 = {"batch_size":tune.choice(range(50,200,10)),
        "encoding":tune.choice(["blosum"]),
        #"encoding":tune.choice(["blosum","onehot"]),
        "likelihood_scale":tune.choice(range(40,100,10)),
        "num_epochs":tune.choice([max_num_epochs]),
        "num_samples":tune.choice([30,40,50,60]),
        "hidden_dim":tune.choice(range(20,45,10)),
        "z_dim": tune.choice([2 * i for i in range(1, 20)]),
        }


    args_dict = vars(args)
    args_dict["config_dict"] = config1
    args_dict = {**args_dict,**config2}
    args = Namespace(**args_dict)


    num_ray_samples = 100

    scheduler = ASHAScheduler(
        metric="valid_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )


    excludes = ["vegvisir/data/{}/*".format(os.path.basename(f.path)) for f in os.scandir("{}/vegvisir/src/vegvisir/data".format(dataset_info.script_dir)) if f.is_dir() and os.path.basename(f.path) not in ["common_files"]]

    excludes2 = ["vegvisir/data/common_files/{}".format(os.path.basename(f.path))for f in os.scandir("{}/vegvisir/src/vegvisir/data/common_files".format(dataset_info.script_dir)) if os.path.basename(f.path) not in ["aminoacid_properties.txt","allele_colors2.tsv","alleles_dict.txt"]]

    ray.init(runtime_env={"working_dir": "{}/vegvisir/src/".format(dataset_info.script_dir),
                          #'excludes':["!/vegvisir/data/common_files/aminoacid_properties.txt","/vegvisir/data/*"], #not working had to do it manually
                          'excludes':excludes+excludes2, #"!/vegvisir/data/common_files"

                          },
            logging_level = logging.DEBUG
             )

    if args.k_folds > 1:
        print("Initializing Hyperparameter Optimization with K-fold cross validation")
        train_fn = VegvisirTrainSVI.kfold_crossvalidation
    else:
        print("Initializing Hyperparameter Optimization for single fold")
        train_fn = VegvisirTrainSVI.train_model
    # Highlight: Find the name of the GPU, seem to start with the accelerator name
    gpu_name = [key if "accelerator" in key else None for key in ray.available_resources().keys()]
    gpu_name = list(filter(partial(is_not, None), gpu_name))
    if gpu_name:
        print("GPU accelerator found!")
        resources_dict = {"cpu":multiprocessing.cpu_count()-10,"gpu":1,gpu_name[0]:1}
    else:
        resources_dict = {"cpu":multiprocessing.cpu_count()-10,"gpu":0}
    result = tune.run(
        tune.with_parameters(train_fn, dataset_info=dataset_info,additional_info=additional_info,args=args),
        #resources_per_trial={"cpu":35,"gpu":1,"accelerator_type:RTX":1},
        resources_per_trial=resources_dict,
        config={**config1,**config2},
        num_samples=num_ray_samples,
        scheduler=scheduler,
        max_failures=0,
        stop={}, #stop if some criteria in the metrics is met
        fail_fast=True
    )

    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['valid_loss']}")
    print(f"Best trial final validation ROC AUC: {best_trial.last_result['ROC_AUC_valid']}")

    best_trial_dict = {**best_trial.last_result,**best_trial.config}
    best_trial_dict.pop("config",None)

    for key,val in best_trial_dict.items():
        if not isinstance(val,str):
            if val is not None:
                best_trial_dict[key] = float(val)


    json.dump(best_trial_dict, open("{}/Best_validation_loss.p".format(additional_info.results_dir), "w+"),indent=2)

    best_trial = result.get_best_trial("ROC_AUC_valid", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['valid_loss']}")
    print(f"Best trial final validation ROC AUC: {best_trial.last_result['ROC_AUC_valid']}")

    best_trial_dict = {**best_trial.last_result, **best_trial.config}
    best_trial_dict.pop("config", None)

    for key, val in best_trial_dict.items():
        if not isinstance(val, str):
            if val is not None:
                best_trial_dict[key] = float(val)

    json.dump(best_trial_dict,open("{}/Best_validation_auc.p".format(additional_info.results_dir),"w+"),indent=2)


    print("-----------------------DONE! saving ... ----------------------")

    result.results_df.to_csv("{}/HyperparamOptimization_results.tsv".format(additional_info.results_dir),sep="\t")


    #TODO: After HPO, keep best configuration and call Vegvisir_example.py with the new configuration


def run(dataset_info,results_dir,args):
    """Execute K-fold cross validation over the processed dataset"""
    additional_info = AdditionalInfo(results_dir=results_dir) #TODO: Simply "update" dataset_info

    if args.hpo:
        hyperparameter_optimization(dataset_info, additional_info, args)
    else:

        assert args.num_synthetic_peptides < 10000, "Please generate less than 10000 peptides, otherwise the computations might not be posible or they might take too long"

        if args.k_folds <= 1:
            VegvisirTrainSVI.train_model(config=None,dataset_info=dataset_info,additional_info=additional_info,args=args)
        else:
            print("Initializing {}-fold cross validation".format(args.k_folds))
            VegvisirTrainSVI.kfold_crossvalidation(config = None, dataset_info=dataset_info,additional_info=additional_info,args=args)


        #VegvisirTrainRF.train_xgboost_binary_classifier(dataset_info,additional_info,args)
        #VegvisirTrainRF.train_xgboost_regression(dataset_info,additional_info,args)
