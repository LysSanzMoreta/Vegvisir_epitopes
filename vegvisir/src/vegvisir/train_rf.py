from xgboost import XGBClassifier
import xgboost as xgb
from vegvisir.train import dataset_proportions,fold_auc
import vegvisir.plots as VegvisirPlots
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
import numpy as np
import pandas as pd
from collections import defaultdict
import dataframe_image as dfi
import seaborn as sns
def trainevaltest_split(data,args,results_dir,method="predefined_partitions"):
    """Perform train-test split"""
    if method == "predefined_partitions":
        #Train - Test split
        traineval_data,test_data = data[data[:,0,3] == 1.], data[data[:,0,3] == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        partitions = traineval_data[:,0,2]
        unique_partitions = np.unique(partitions)
        i=1
        kfolds = []
        for part_num in unique_partitions:
            #train_idx = traineval_data[traineval_data[:,0,2] != part_num]
            train_idx = (traineval_data[:,0,2][..., None] != part_num).any(-1)
            valid_idx = (traineval_data[:,0,2][..., None] == part_num).any(-1)
            kfolds.append((train_idx,valid_idx))
            if args.k_folds <= i :
                break
            else:
                i+=1
        return traineval_data,test_data,kfolds

    elif method == "stratified_group_partitions":
        #Train - Test split
        traineval_data,test_data = data[data[:,0,3] == 1.], data[data[:,0,3] == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        partitions = traineval_data[:,0,2]
        #Train - Eval split
        kfolds = StratifiedGroupKFold(n_splits=args.k_folds).split(traineval_data, traineval_data[:,0,0], partitions)
        return traineval_data,test_data,kfolds
    elif method == "random_stratified":
        data_labels = data[:,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir, type="Test")
        # Train - Eval split
        kfolds = StratifiedShuffleSplit(n_splits=args.k_folds, random_state=13, test_size=0.2).split(traineval_data,traineval_data[:,0,0])
        return traineval_data,test_data,kfolds
    else:
        raise ValueError("train test split method not available")
def train_xgboost(dataset_info,additional_info,args):

    data_blosum_norm = dataset_info.data_array_blosum_norm
    results_dir = additional_info.results_dir
    traineval_data_blosum,test_data_blosum,kfolds = trainevaltest_split(data_blosum_norm,args,results_dir,method="predefined_partitions")

    auc_dict = defaultdict(lambda : defaultdict(list))
    auk_dict = defaultdict(lambda : defaultdict(list))
    feature_dict = defaultdict(list)
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation
        print("Running fold {} ......".format(fold))
        train_data_blosum = traineval_data_blosum[train_idx]
        eval_data_blosum = traineval_data_blosum[valid_idx]
        print("\t Total number of data points: {}".format(traineval_data_blosum.shape[0]))
        print('\t Number train data points: {}; Proportion: {}'.format(len(train_data_blosum), (len(train_data_blosum) * 100) /
                                                                       traineval_data_blosum.shape[0]))
        print('\t Number valid data points: {}; Proportion: {}'.format(len(eval_data_blosum), (len(eval_data_blosum) * 100) /
                                                                       traineval_data_blosum.shape[0]))
        dataset_proportions(train_data_blosum,results_dir)
        dataset_proportions(eval_data_blosum,results_dir,type="Valid")

        # create model instance
        xgbc = XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.01, objective='binary:logistic',tree_method="auto")
        # fit model
        xgbc.fit(train_data_blosum[:,1:].squeeze(1), train_data_blosum[:,0,0])
        # make predictions
        preds_train = xgbc.predict(train_data_blosum[:,1:].squeeze(1))
        preds_eval = xgbc.predict(eval_data_blosum[:,1:].squeeze(1))
        auc_score_train,auk_score_train=fold_auc(preds_train,train_data_blosum[:,0,0],fold,results_dir,mode="Train")
        auc_dict["Fold_{}".format(fold)]["Train"] = auc_score_train
        auk_dict["Fold_{}".format(fold)]["Train"] = auk_score_train

        auc_score_valid,auk_score_valid=fold_auc(preds_eval,eval_data_blosum[:,0,0],fold,results_dir,mode="Valid")
        auc_dict["Fold_{}".format(fold)]["Valid"] = auc_score_valid
        auk_dict["Fold_{}".format(fold)]["Valid"] = auk_score_valid

        print("--------------------------------------------")
        feature_dict["Fold_{}".format(fold)] = xgbc.feature_importances_


    print("Running with entire dataset and final testing ")
    # create model instance
    xgbc = XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.01, objective='binary:logistic',
                        tree_method="auto")
    # fit model
    xgbc.fit(traineval_data_blosum[:, 1:].squeeze(1), traineval_data_blosum[:, 0, 0])
    # make predictions
    preds_train = xgbc.predict(traineval_data_blosum[:, 1:].squeeze(1))
    preds_test = xgbc.predict(test_data_blosum[:, 1:].squeeze(1))
    # Cross validation: https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html
    auc_score_train, auk_score_train=fold_auc(preds_train, traineval_data_blosum[:, 0, 0], "all", results_dir, mode="Train")
    auc_dict["Full_dataset"]["Train"] = auc_score_train
    auk_dict["Full_dataset"]["Train"] = auk_score_train
    auc_score_test, auk_score_test=fold_auc(preds_test, test_data_blosum[:, 0, 0], "all", results_dir, mode="Test")
    auc_dict["Full_dataset"]["Test"] = auc_score_test
    auk_dict["Full_dataset"]["Test"] = auk_score_test
    feature_dict["Full_dataset"] = xgbc.feature_importances_

    auc_df = pd.DataFrame.from_dict(auc_dict)
    #auc_df_styled = auc_df.style.background_gradient(axis=None).format(na_rep = "0") #cmap="BuPu"
    auc_df_styled = auc_df.style.format(na_rep = "-") #cmap="BuPu"
    dfi.export(auc_df_styled, "{}/AUC_df.png".format(results_dir), dpi=600)
    VegvisirPlots.plot_feature_importance(feature_dict, results_dir)


