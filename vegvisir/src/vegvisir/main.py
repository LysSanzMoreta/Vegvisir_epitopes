"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import vegvisir
import vegvisir.train as VegvisirTrain
import vegvisir.train_svi as VegvisirTrainSVI
import vegvisir.train_rf as VegvisirTrainRF
from collections import namedtuple
AdditionalInfo = namedtuple("AdditionalInfo",["results_dir"])

def run(dataset_info,results_dir,args):
    """Execute K-fold cross validation over the processed dataset"""
    additional_info = AdditionalInfo(results_dir=results_dir)
    if args.run_nnalign:
        print("Done running NNAlign ....")
    else:
        #VegvisirTrain.kfold_crossvalidation(dataset_info,additional_info,args)
        #VegvisirTrain.train_model(dataset_info,additional_info,args) #ordinary train,val,test split without k-fold cross validation.The validation set changes every time

        #VegvisirTrainSVI.kfold_crossvalidation(dataset_info,additional_info,args)
        VegvisirTrainSVI.train_model(dataset_info,additional_info,args)

        #VegvisirTrainRF.train_xgboost_binary_classifier(dataset_info,additional_info,args)
        #VegvisirTrainRF.train_xgboost_regression(dataset_info,additional_info,args)
