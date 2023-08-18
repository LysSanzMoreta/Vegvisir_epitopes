"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import math
from argparse import Namespace

import numpy as np

import vegvisir
import vegvisir.train as VegvisirTrain
import vegvisir.train_svi as VegvisirTrainSVI
import vegvisir.train_rf as VegvisirTrainRF
from collections import namedtuple
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from functools import partial
AdditionalInfo = namedtuple("AdditionalInfo",["results_dir"])


def hyperparameter_optimization(dataset_info,additional_info,args):
    """Initiates Hyperparameter search with Ray-tune
    Notes:
        -Adam parameters: https://www.kdnuggets.com/2022/12/tuning-adam-optimizer-parameters-pytorch.html
    """

    config1 = {
        "lr":tune.choice([1e-4,1e-3,1e-2]),
        "beta1": tune.choice([0.9,0.91,0.92,0.93,0.94]),
        "beta2": tune.choice([0.999,0.9999]),
        "eps": tune.choice([1e-8]), #not tuning parameter, avoids division by 0
        "weight_decay": tune.choice([0]),
        "clip_norm":tune.choice([10]), #not used with pyro
        "lrd":tune.choice([1]),
        "z_dim":tune.choice([2*i for i in range(1,20)]),
        "gru_hidden_dim":tune.choice(np.arange(50,90,10)),
        "momentum":tune.randn(0.9,0.1)
    }
    config2 = {"batch_size":tune.choice(np.arange(50,200,10)),
        "encoding":tune.choice(["blosum","onehot"]),
        "likelihood_scale":tune.choice(np.arange(40,100,10)),
        "num_epochs":tune.choice([2]),
        "num_samples":tune.choice([30,40,50]),
        "hidden_dim":tune.choice(np.arange(20,45,10)),
        }


    args_dict = vars(args)
    args_dict["config_dict"] = config1
    args_dict = {**args_dict,**config2}
    args = Namespace(**args_dict)

    num_ray_samples = 5
    max_num_epochs = 30
    scheduler = ASHAScheduler(
        metric="valid_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )


    ray.init(runtime_env={"working_dir": "{}/vegvisir/src".format(dataset_info.script_dir),'excludes':["/vegvisir/data/"]})

    #TODO: Stopper: https://stackoverflow.com/questions/71134673/python-ray-tune-unable-to-stop-trial-or-experiment

    result = tune.run(
        tune.with_parameters(VegvisirTrainSVI.train_model, dataset_info=dataset_info,additional_info=additional_info,args=args),
        resources_per_trial={"cpu":30,"gpu":1,"accelerator_type:RTX":1},
        config={**config1,**config2},
        num_samples=num_ray_samples,
        scheduler=scheduler,
        max_failures=0,
        stop={}, #stop if some criteria in the metrics is met
        fail_fast=True
    )

    print(result)
    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['valid_loss']}")
    print(f"Best trial final validation ROC AUC: {best_trial.last_result['ROC_AUC_valid']}")

    best_trial = result.get_best_trial("ROC_AUC_valid", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['valid_loss']}")
    print(f"Best trial final validation ROC AUC: {best_trial.last_result['ROC_AUC_valid']}")



def run(dataset_info,results_dir,args):
    """Execute K-fold cross validation over the processed dataset"""
    additional_info = AdditionalInfo(results_dir=results_dir)
    if args.run_nnalign:
        print("Done running NNAlign ....")

    elif args.hpo:
        if args.k_folds <= 1:
            hyperparameter_optimization(dataset_info,additional_info,args)
        else:
            print("Hyperparameter optimization not implemented for args.k_folds > 1")
    else:
        #VegvisirTrain.kfold_crossvalidation(dataset_info,additional_info,args)
        #VegvisirTrain.train_model(dataset_info,additional_info,args) #ordinary train,val,test split without k-fold cross validation.The validation set changes every time

        #VegvisirTrainSVI.kfold_crossvalidation(dataset_info,additional_info,args)


        # if args.pretrained_model is not None:
        #     print("Loading pre-trained model from {}".format(args.pretrained_model))
        #     VegvisirTrainSVI.load_model(dataset_info,additional_info,args)
        #
        # else:
        if args.k_folds <= 1:
            VegvisirTrainSVI.train_model(config=None,dataset_info=dataset_info,additional_info=additional_info,args=args)
        else:
            print("Initializing {}-fold cross validation".format(args.k_folds))
            VegvisirTrainSVI.kfold_crossvalidation(dataset_info,additional_info,args)


        #VegvisirTrainRF.train_xgboost_binary_classifier(dataset_info,additional_info,args)
        #VegvisirTrainRF.train_xgboost_regression(dataset_info,additional_info,args)
