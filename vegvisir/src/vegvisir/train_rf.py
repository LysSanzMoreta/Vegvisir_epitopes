from xgboost import XGBClassifier
from vegvisir.train import dataset_proportions,fold_auc
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold

def trainevaltest_split(data,args,results_dir,method="predefined_partitions"):
    """Perform train-test split"""
    if method == "predefined_partitions":
        #Train - Test split
        traineval_data,test_data = data[data[:,0,3] == 1.], data[data[:,0,3] == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        #Train - Eval split
        kfolds = StratifiedGroupKFold(n_splits=args.k_folds).split(traineval_data, traineval_data[:,0,0], traineval_data[:,0,2])
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

    # create model instance
    bst = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.01, objective='binary:logistic')
    # fit model
    bst.fit(traineval_data_blosum[:,1:].squeeze(1), traineval_data_blosum[:,0,0])
    # make predictions
    preds = bst.predict(test_data_blosum[:,1:].squeeze(1))


    fold_auc(preds,test_data_blosum[:,0,0],0,results_dir,mode="Test")