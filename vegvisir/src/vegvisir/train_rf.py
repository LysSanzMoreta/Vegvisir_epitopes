from xgboost import XGBClassifier
import xgboost as xgb
from vegvisir.train import dataset_proportions,fold_auc
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
import numpy as np
def trainevaltest_split(data,args,results_dir,method="predefined_partitions"):
    """Perform train-test split"""
    if method == "predefined_partitions":
        #Train - Test split
        traineval_data,test_data = data[data[:,0,3] == 1.], data[data[:,0,3] == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        partitions = traineval_data[:,0,2]
        unique_partitions = np.unique(partitions)
        kfolds = []
        for part_num in unique_partitions:
            #train_idx = traineval_data[traineval_data[:,0,2] != part_num]
            train_idx = (traineval_data[:,0,2][..., None] != part_num).any(-1)
            valid_idx = (traineval_data[:,0,2][..., None] == part_num).any(-1)
            kfolds.append((train_idx,valid_idx))
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
        #Cross validation: https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html
        fold_auc(preds_train,train_data_blosum[:,0,0],fold,results_dir,mode="Train")
        fold_auc(preds_eval,eval_data_blosum[:,0,0],fold,results_dir,mode="Valid")
        print("--------------------------------------------")


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
    fold_auc(preds_train, traineval_data_blosum[:, 0, 0], "all", results_dir, mode="Train")
    fold_auc(preds_test, test_data_blosum[:, 0, 0], "all", results_dir, mode="Test")

    # def fpreproc(dtrain, dtest, param):
    #     label = dtrain[:,0,0]
    #     ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    #     param['scale_pos_weight'] = ratio
    #     return (dtrain, dtest, param)
    #
    # xgb.cv({"n_estimators":1500,'max_depth':8, 'eta':1, 'objective':'binary:logistic'}, traineval_data_blosum, 2, nfold=5,metrics='auc', seed=0, fpreproc=fpreproc)

