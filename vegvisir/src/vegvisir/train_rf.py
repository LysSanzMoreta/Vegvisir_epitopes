from xgboost import XGBClassifier,XGBRegressor
import xgboost as xgb
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.utils as VegvisirUtils
import vegvisir.plots as VegvisirPlots
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import dataframe_image as dfi
import seaborn as sns
from sklearn.metrics import r2_score,mean_absolute_error
def regression_accuracy(predictions,targets,fold,mode):
    """
    :param predictions:
    :param targets:
    :param fold:
    :param mode:
    :return:
    """
    #R-Squared Score
    r2score = r2_score(targets, predictions)
    print("Fold {} ---------------------------------------------------".format(fold))
    print("{}. The R2score accuracy of our model is {}%".format(mode,round(r2score, 2) * 100))
    mse_error = mean_absolute_error(targets, predictions)
    print("{}. The Mean Absolute Error of our Model is {}".format(mode,round(mse_error, 2)))
    rmse_score = np.sqrt(mse_error)
    print("{}. The Root Mean Absolute Error of our Model is {}".format(mode,round(rmse_score, 2)))
    return round(r2score, 2) * 100,round(mse_error, 2),round(rmse_score, 2)


def train_xgboost_binary_classifier(dataset_info,additional_info,args):
    """

    :param dataset_info:
    :param additional_info:
    :param args:
    """
    data_blosum_norm = dataset_info.data_array_blosum_norm
    results_dir = additional_info.results_dir
    max_len = dataset_info.max_len
    features_names = dataset_info.features_names
    if features_names is not None:
        feature_names = ["Pos.{}".format(pos) for pos in list(range(max_len))] + features_names
    else:
        feature_names = ["Pos.{}".format(pos) for pos in list(range(max_len))]

    VegvisirPlots.plot_mutual_information(data_blosum_norm,data_blosum_norm[:,0,0],feature_names,results_dir)

    traineval_data_blosum,test_data_blosum,kfolds = VegvisirLoadUtils.trainevaltest_split_kfolds(data_blosum_norm,args,results_dir,method="predefined_partitions")

    auc_dict = defaultdict(lambda : defaultdict(list))
    auk_dict = defaultdict(lambda : defaultdict(list))
    feature_dict = defaultdict(list)
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation
        print("Running fold {} ......".format(fold))
        train_data_blosum = traineval_data_blosum[train_idx]
        valid_data_blosum = traineval_data_blosum[valid_idx]
        print("\t Total number of data points: {}".format(traineval_data_blosum.shape[0]))
        print('\t Number train data points: {}; Proportion: {}'.format(len(train_data_blosum), (len(train_data_blosum) * 100) /
                                                                       traineval_data_blosum.shape[0]))
        print('\t Number valid data points: {}; Proportion: {}'.format(len(valid_data_blosum), (len(valid_data_blosum) * 100) /
                                                                       traineval_data_blosum.shape[0]))
        VegvisirLoadUtils.dataset_proportions(train_data_blosum,results_dir)
        VegvisirLoadUtils.dataset_proportions(valid_data_blosum,results_dir,type="Valid")

        # create model instance
        xgbc = XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.01, objective='binary:logistic',tree_method="auto")
        # fit model
        xgbc.fit(train_data_blosum[:,1:].squeeze(1), train_data_blosum[:,0,0])
        # make predictions
        preds_train = xgbc.predict(train_data_blosum[:,1:].squeeze(1))
        preds_valid = xgbc.predict(valid_data_blosum[:,1:].squeeze(1))
        accuracy_train = (torch.Tensor(preds_train).cpu() == train_data_blosum[:,0,0]).sum()/preds_train.shape[0]
        auc_score_train,auk_score_train=VegvisirUtils.fold_auc(preds_train,train_data_blosum[:,0,0],accuracy_train,fold,results_dir,mode="Train")
        auc_dict["Fold_{}".format(fold)]["Train"] = auc_score_train
        auk_dict["Fold_{}".format(fold)]["Train"] = auk_score_train
        accuracy_valid = (torch.Tensor(preds_valid).cpu() == valid_data_blosum[:,0,0]).sum()/preds_train.shape[0]
        auc_score_valid,auk_score_valid=VegvisirUtils.fold_auc(preds_valid,valid_data_blosum[:,0,0],accuracy_valid,fold,results_dir,mode="Valid")
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
    accuracy_train = (torch.Tensor(preds_train).cpu() == traineval_data_blosum[:, 0, 0]).sum() / preds_train.shape[0]
    auc_score_train, auk_score_train=VegvisirUtils.fold_auc(preds_train, traineval_data_blosum[:, 0, 0],accuracy_train, "all", results_dir, mode="Train")
    auc_dict["Full_dataset"]["Train"] = auc_score_train
    auk_dict["Full_dataset"]["Train"] = auk_score_train
    accuracy_test = (torch.Tensor(preds_test).cpu() == test_data_blosum[:, 0, 0]).sum() / preds_test.shape[0]
    auc_score_test, auk_score_test=VegvisirUtils.fold_auc(preds_test, test_data_blosum[:, 0, 0], accuracy_test,"all", results_dir, mode="Test")
    auc_dict["Full_dataset"]["Test"] = auc_score_test
    auk_dict["Full_dataset"]["Test"] = auk_score_test
    feature_dict["Full_dataset"] = xgbc.feature_importances_


    auc_df = pd.DataFrame.from_dict(auc_dict).transpose()
    auc_df_styled = auc_df.style.background_gradient(axis=None).format(na_rep = "0") #cmap="BuPu"
    auc_df_styled = auc_df.style.format(na_rep = "-") #cmap="BuPu"
    #auc_df_styled.to_html()
    dfi.export(auc_df_styled, "{}/AUC_df.png".format(results_dir), dpi=600)
    VegvisirPlots.plot_feature_importance(feature_dict, max_len,features_names,results_dir)

def train_xgboost_regression(dataset_info,additional_info,args):
    """
    Notes:
        - https://towardsdatascience.com/xgboost-regression-explain-it-to-me-like-im-10-2cf324b0bbdb
    :param dataset_info:
    :param additional_info:
    :param args:
    """
    data_blosum_norm = dataset_info.data_array_blosum_norm
    results_dir = additional_info.results_dir
    max_len = dataset_info.max_len
    features_names = dataset_info.features_names
    if features_names is not None:
        feature_names = ["Pos.{}".format(pos) for pos in list(range(max_len))] + features_names
    else:
        feature_names = ["Pos.{}".format(pos) for pos in list(range(max_len))]

    VegvisirPlots.plot_mutual_information(data_blosum_norm,data_blosum_norm[:,0,0],feature_names,results_dir)

    traineval_data_blosum,test_data_blosum,kfolds = VegvisirLoadUtils.trainevaltest_split_kfolds(data_blosum_norm,args,results_dir,method="predefined_partitions")

    rs2score_dict = defaultdict(lambda : defaultdict(list))
    rmse_dict = defaultdict(lambda : defaultdict(list))
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
        VegvisirLoadUtils.dataset_proportions(train_data_blosum,results_dir)
        VegvisirLoadUtils.dataset_proportions(eval_data_blosum,results_dir,type="Valid")

        # create model instance
        xgbc = XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.01, objective='reg:logistic',tree_method="auto")
        # fit model
        xgbc.fit(train_data_blosum[:,1:].squeeze(1), train_data_blosum[:,0,0])
        # make predictions
        preds_train = xgbc.predict(train_data_blosum[:,1:].squeeze(1))
        preds_eval = xgbc.predict(eval_data_blosum[:,1:].squeeze(1))
        r2score_train,mse_error_train,rmse_score_train=regression_accuracy(preds_train,train_data_blosum[:,0,4],fold,mode="Train")
        rs2score_dict["Fold_{}".format(fold)]["Train"] = r2score_train
        rmse_dict["Fold_{}".format(fold)]["Train"] = rmse_score_train
        r2score_valid,mse_error_valid,rmse_score_valid=regression_accuracy(preds_eval,eval_data_blosum[:,0,4],fold,mode="Valid")
        rs2score_dict["Fold_{}".format(fold)]["Valid"] = r2score_valid
        rmse_dict["Fold_{}".format(fold)]["Valid"] = rmse_score_valid

        print("--------------------------------------------")
        feature_dict["Fold_{}".format(fold)] = xgbc.feature_importances_


    print("Running with entire dataset and final testing ")
    # create model instance
    xgbc = XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.01, objective='reg:logistic',tree_method="auto")
    # fit model
    xgbc.fit(traineval_data_blosum[:, 1:].squeeze(1), traineval_data_blosum[:, 0, 4])
    # make predictions
    preds_train = xgbc.predict(traineval_data_blosum[:, 1:].squeeze(1))
    preds_test = xgbc.predict(test_data_blosum[:, 1:].squeeze(1))
    # Cross validation: https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html
    r2score_train,mse_error_train,rmse_score_train=regression_accuracy(preds_train, traineval_data_blosum[:, 0, 4], "all", mode="Train")
    rs2score_dict["Full_dataset"]["Train"] = r2score_train
    rmse_dict["Full_dataset"]["Train"] = rmse_score_train
    r2score_test,mse_error_test,rmse_score_test=regression_accuracy(preds_test, test_data_blosum[:, 0, 4], "all", mode="Test")
    rs2score_dict["Full_dataset"]["Test"] = r2score_test
    rmse_dict["Full_dataset"]["Test"] = rmse_score_test
    feature_dict["Full_dataset"] = xgbc.feature_importances_


    r2score_df = pd.DataFrame.from_dict(rs2score_dict).transpose()
    #r2score_df_styled = r2score_df.style.background_gradient(axis=None).format(na_rep = "0") #cmap="BuPu"
    r2score_df_styled = r2score_df.style.format(na_rep = "-") #cmap="BuPu"
    #r2score_df_styled.to_html()
    dfi.export(r2score_df_styled, "{}/R2Score_df.png".format(results_dir), dpi=600)

    rmse_df = pd.DataFrame.from_dict(rmse_dict).transpose()
    rmse_df_styled = rmse_df.style.format(na_rep="-")  # cmap="BuPu"
    # r2score_df_styled.to_html()
    dfi.export(rmse_df_styled, "{}/RMSE_df.png".format(results_dir), dpi=600)

    VegvisirPlots.plot_feature_importance(feature_dict, max_len,features_names,results_dir)



