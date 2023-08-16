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
    """Initiates Hyperparameter search with Ray-tune"""
    num_samples = 1
    max_num_epochs = 30
    gpus_per_trial = 1

    config1 = {
        "lr":tune.choice([1e-4,1e-3,1e-2]),
        "beta1": tune.randn(0.95,0.05),
        "beta2": tune.randn(0.999,0.001),
        "eps": tune.choice([1e-8, 1e-7,1e-6]),
        "weight_decay": tune.randn(0,0.05),
        "clip_norm":tune.randn(10,2),
        "lrd":tune.randn(1,2),
        "z_dim":tune.choice([2*i for i in range(20)]),
        "gru_hidden_dim":tune.choice(np.arange(50,90,10)),
        "momentum":tune.randn(0.9,0.1)
    }
    config2 = {"batch_size":tune.choice(np.arange(50,200,10)),
        "encoding":tune.choice(["blosum","onehot"]),
        "likelihood_scale":tune.choice(np.arange(40,100,10)),
        "num_epochs":tune.choice([2]),
        "hidden_dim":tune.choice(np.arange(20,45,10)),
        }


    args_dict = vars(args)
    args_dict["config_dict"] = config1
    args_dict = {**args_dict,**config2}
    args = Namespace(**args_dict)

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
        num_samples=num_samples,
        scheduler=scheduler,
        max_failures=0,
        stop={}
    )

    print(result)
    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)
    #
    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    # best_checkpoint_data = best_checkpoint.to_dict()
    #
    # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    #
    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


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
