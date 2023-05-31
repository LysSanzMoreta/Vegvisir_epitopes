"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import random
import warnings
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
import torch
import vegvisir.utils as VegvisirUtils
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots

class CustomDataset(Dataset):
    def __init__(self, data_array_blosum,data_array_int,data_array_onehot,data_array_blosum_norm,batch_mask,batch_positional_mask):
        self.batch_data_blosum = data_array_blosum
        self.batch_data_int = data_array_int
        self.batch_data_onehot = data_array_onehot
        self.batch_data_blosum_norm = data_array_blosum_norm
        self.batch_mask = batch_mask
        self.batch_positional_mask = batch_positional_mask
    def __getitem__(self, index):  # sets a[i]
        batch_data_blosum = self.batch_data_blosum[index]
        batch_data_int = self.batch_data_int[index]
        batch_data_onehot = self.batch_data_onehot[index]
        batch_data_blosum_norm = self.batch_data_blosum_norm[index]
        batch_mask = self.batch_mask[index]
        batch_positional_mask = self.batch_positional_mask[index]
        return {'batch_data_blosum': batch_data_blosum,
                'batch_data_int':batch_data_int,
                'batch_data_onehot':batch_data_onehot,
                'batch_data_blosum_norm':batch_data_blosum_norm,
                'batch_mask':batch_mask,
                'batch_positional_mask':batch_positional_mask}
    def __len__(self):
        return len(self.batch_data_blosum)

def dataset_proportions(data,results_dir,type="TrainEval"):
    """Calculates distribution of data points based on their labeling"""
    if isinstance(data,np.ndarray):
        data = torch.from_numpy(data)
    if data.ndim == 4:
        positives = torch.sum(data[:,0,0,0])
    else:
        positives = torch.sum(data[:,0,0])
    positives_proportion = (positives*100)/torch.tensor([data.shape[0]])
    negatives = data.shape[0] - positives
    negatives_proportion = 100-positives_proportion
    f = open("{}/dataset_info.txt".format(results_dir),"a+")
    print("\n{} dataset: \n \t Total number of data points: {} \n \t Number positives : {}; \n \t Proportion positives : {} ; \n \t Number negatives : {} ; \n \t Proportion negatives : {}".format(type,data.shape[0],positives,positives_proportion.item(),negatives,negatives_proportion.item()))
    print("-------------------------------------------------------------",file=f)
    print("\n{} dataset: \n \t Total number of data points: {} \n \t Number positives : {}; \n \t Proportion positives : {} ; \n \t Number negatives : {} ; \n \t Proportion negatives : {}".format(type,data.shape[0],positives,positives_proportion.item(),negatives,negatives_proportion.item()),file=f)
    return (positives,positives_proportion),(negatives,negatives_proportion)

def trainevaltest_split_kfolds(data,args,results_dir,method="predefined_partitions"):
    """Perform kfolds partitions division and test split"""

    warnings.warn("Feature Scaling has not been implemented for k-fold cross validation, please remember to do so")
    if data.ndim == 4: #TODO:not sure why the list comprehesion does not work with lambda
        idx_select = lambda x,n: x[:,0,0,n] #TODO: Use ellipsis? so far not working, this is best
    else:
        idx_select = lambda x,n: x[:,0,n]

    if method == "predefined_partitions":
        #traineval_data, test_data = data[data[:, 0,0, 3] == 1.], data[data[:, 0,0, 3] == 0.]
        traineval_data, test_data = data[idx_select(data,3) == 1.], data[idx_select(data,3) == 0.]

        dataset_proportions(traineval_data, results_dir)
        dataset_proportions(test_data, results_dir, type="Test")
        #partitions = traineval_data[:, 0,0, 2]
        partitions = idx_select(traineval_data,2)
        unique_partitions = np.unique(partitions)
        assert args.k_folds <= len(unique_partitions), "kfold number is too high, please select a number lower than {}".format(len(unique_partitions))
        i = 1
        kfolds = []
        for part_num in unique_partitions:
            #train_idx = (traineval_data[:, 0,0, 2][..., None] != part_num).any(-1)
            train_idx = (idx_select(traineval_data,2)[..., None] != part_num).any(-1)
            #valid_idx = (traineval_data[:, 0,0, 2][..., None] == part_num).any(-1)
            valid_idx = (idx_select(traineval_data,2)[..., None] == part_num).any(-1)
            kfolds.append((train_idx, valid_idx))
            if args.k_folds <= i :
                break
            else:
                i+=1
        return traineval_data, test_data, kfolds
    elif method == "stratified_group_partitions":
        traineval_data,test_data = data[idx_select(data,3) == 1.], data[idx_select(data,3) == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        kfolds = StratifiedGroupKFold(n_splits=args.k_folds).split(traineval_data, idx_select(traineval_data,0), idx_select(traineval_data,2))
        return traineval_data,test_data,kfolds
    elif method == "random_stratified":
        data_labels = idx_select(data,0)
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir, type="Test")
        kfolds = StratifiedShuffleSplit(n_splits=args.k_folds, random_state=13, test_size=0.2).split(traineval_data,idx_select(traineval_data,0))
        return traineval_data,test_data,kfolds
    elif method == "predefined_partitions_discard_test":
        """Discard the test dataset (hard case) and use one of the partitions as the test instead. The rest of the dataset is used for the kfold partitions"""
        partition_idx = np.random.randint(0, 4)  # random selection of a partition as the test
        train_data = data[idx_select(data,2) != partition_idx]
        test_data = data[idx_select(data,2) == partition_idx]  # data[data[:, 0, 0, 3] == 1.],
        dataset_proportions(train_data, results_dir)
        dataset_proportions(test_data, results_dir, type="Test")
        partitions = idx_select(train_data,2)
        unique_partitions = np.unique(partitions)
        assert args.kfolds <= len(unique_partitions), "kfold number is too high, please select a number lower than {}".format(len(unique_partitions))
        i = 1
        kfolds = []
        for part_num in unique_partitions:
            # train_idx = traineval_data[traineval_data[:,0,2] != part_num]
            train_idx = (idx_select(train_data,2)[..., None] != part_num).any(-1)
            valid_idx = (idx_select(train_data,2)[..., None] == part_num).any(-1)
            kfolds.append((train_idx, valid_idx))
            if args.k_folds <= i:
                break
            else:
                i += 1
        return train_data, test_data, kfolds

    else:
        raise ValueError("train test split method <{}> not available".format(method))

def trainevaltest_split(data,args,results_dir,seq_max_len,max_len,features_names,partition_test,method="predefined_partitions_discard_test"):
    """Perform train-valid-test split"""
    info_file = open("{}/dataset_info.txt".format(results_dir),"a+")
    if method == "random_stratified":
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(data, test_size=0.1, random_state=13, stratify=traineval_labels,shuffle=True)
        dataset_proportions(train_data,results_dir, type="Train")
        dataset_proportions(valid_data,results_dir, type="Valid")
        dataset_proportions(test_data,results_dir, type="Test")
    elif method == "random_stratified_discard_test":
        """Discard the predefined test dataset"""
        data = data[data[:,0,0,3] == 1] #pick only the pre assigned training data
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(traineval_data, test_size=0.1, random_state=13, stratify=traineval_labels,shuffle=True)
        dataset_proportions(train_data,results_dir, type="Train")
        dataset_proportions(valid_data,results_dir, type="Valid")
        dataset_proportions(test_data,results_dir, type="Test")
    elif method == "predefined_partitions_discard_test":
        """Discard the test dataset (because it seems a hard case) and use one of the partitions as the test instead. 
        The rest of the dataset is used for the training, and a portion for validation"""
        if partition_test is not None:
            partition_idx = partition_test
        else:
            partition_idx = np.random.randint(0,4) #random selection of a partition as the test
        traineval_data = data[data[:, 0, 0, 2] != partition_idx]
        test_data = data[data[:, 0, 0, 2] == partition_idx] #data[data[:, 0, 0, 3] == 1.]
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(traineval_data, test_size=0.1, random_state=13,stratify=traineval_labels, shuffle=True)
        dataset_proportions(train_data, results_dir, type="Train")
        dataset_proportions(valid_data, results_dir, type="Valid")
        dataset_proportions(test_data, results_dir, type="Test")
        info_file.write("\n -------------------------------------------------")
        info_file.write("\n Using as test partition: {}".format(partition_idx))
    elif method == "predefined_partitions_no_test":
        """Diffuse the test dataset to the train and validation datasets"""
        if partition_test is not None:
            partition_idx = partition_test
        else:
            partition_idx = np.random.randint(0,4) #random selection of a partition as the test
        traineval_data = data[data[:, 0, 0, 2] != partition_idx]
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(traineval_data, test_size=0.1, random_state=13,stratify=traineval_labels, shuffle=True)
        test_data = valid_data #the test dataset will be the validation again, since the test has been merged onto the train and validation
        dataset_proportions(train_data, results_dir, type="Train")
        dataset_proportions(valid_data, results_dir, type="Valid")
        dataset_proportions(test_data, results_dir, type="Test")
        info_file.write("\n -------------------------------------------------")
        info_file.write("\n Using as test partition: {}".format(partition_idx))
        warnings.warn("Test dataset == Valid dataset, since it has been diffused onto the train and the validation datasets")
    elif method == "predefined_partitions":
        """Use the test data"""

        traineval_data = data[data[:, 0, 0, 3] == 1]
        test_data = data[data[:, 0, 0, 3] == 0] #data[data[:, 0, 0, 3] == 1.]
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(traineval_data, test_size=0.1, random_state=13,stratify=traineval_labels, shuffle=True)
        dataset_proportions(train_data, results_dir, type="Train")
        dataset_proportions(valid_data, results_dir, type="Valid")
        dataset_proportions(test_data, results_dir, type="Test")
        info_file.write("\n -------------------------------------------------")
        info_file.write("\n Using as predefined test")
    else:
        raise ValueError("train test split method not available")

    if seq_max_len != max_len:
        print("Feature preprocessing for train/valid/test splits")
        train_feats,scaler =  VegvisirUtils.features_preprocessing(train_data[:,1,seq_max_len:,0],method="minmax") #TODO: Do after train/valid/test split
        valid_feats = scaler.transform(valid_data[:,1,seq_max_len:,0])
        test_feats = scaler.transform(test_data[:,1,seq_max_len:,0])
        train_data[:,1, seq_max_len:, 0] = torch.from_numpy(train_feats)
        valid_data[:,1, seq_max_len:, 0] = torch.from_numpy(valid_feats)
        test_data[:,1, seq_max_len:, 0] = torch.from_numpy(test_feats)
        VegvisirPlots.plot_features_histogram(train_data, features_names, "{}/Train".format(results_dir), "preprocessed")
        VegvisirPlots.plot_features_histogram(valid_data, features_names, "{}/Valid".format(results_dir), "preprocessed")
        VegvisirPlots.plot_features_histogram(test_data, features_names, "{}/Test".format(results_dir), "preprocessed")

    return train_data,valid_data,test_data

def calculate_class_weights(data,args):
    """Implemented as in https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
    n_samples = data.shape[0]
    y = data[:,0,0,0]
    if isinstance(data,np.ndarray):
        class_weights = n_samples / (args.num_classes * np.bincount(y,minlength=args.num_classes))
    else:
        class_weights = n_samples / (args.num_classes * torch.bincount(y.long(),minlength=args.num_classes))

    return class_weights.to(args.device)

class SequencePadding(object):
    """Performs padding of a list of given sequences to a given len"""
    def __init__(self,sequences,seq_max_len,method,shuffle):
        self.sequences = sequences
        self.seq_max_len = seq_max_len
        self.method = method
        self.shuffle = shuffle
        self.random_seeds = list(range(len(sequences)))

    def run(self):

        padded_sequences = {"no_padding": list(map(lambda seq,seed: self.no_padding(seq,seed, self.seq_max_len,self.shuffle),self.sequences,self.random_seeds)),
                            #"ends":list(map(lambda seq: (list(seq.ljust(self.seq_max_len, "#")),list(seq.ljust(self.seq_max_len, "#"))),self.sequences)) ,
                            "ends": list(map(lambda seq,seed: self.ends_padding(seq,seed, self.seq_max_len,self.shuffle),self.sequences,self.random_seeds)),
                            #"random":list(map(lambda seq,seed: self.random_padding(seq,seed, self.seq_max_len,self.shuffle), self.sequences,self.random_seeds)),
                            #"borders":list(map(lambda seq,seed: self.border_padding(seq,seed, self.seq_max_len,self.shuffle), self.sequences,self.random_seeds)),
                            #"replicated_borders":list(map(lambda seq,seed: self.replicated_border_padding(seq,seed, self.seq_max_len,self.shuffle), self.sequences,self.random_seeds))
                            }



        if self.method not in padded_sequences.keys():
            raise ValueError("Padding method <{}> not implemented, please choose among <{}>".format(self.method,padded_sequences.keys()))
        else:
            return padded_sequences[self.method]

    def no_padding(self,seq,seed,max_len,shuffle):
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        return (random.sample(seq, len(seq)),random.sample(seq, len(seq)))

    def ends_padding(self,seq,seed,max_len,shuffle):
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        return (list(seq.ljust(max_len, "#")),list(seq.ljust(max_len, "#")))

    def random_padding(self,seq,seed, max_len,shuffle):
        """Randomly pad sequence. Introduces <n pads> in random places until max_len"""
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            idx = np.array(random.sample(range(0, max_len), pad), dtype=int)
            new_seq = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx] = False
            new_seq[mask] = np.array(seq)
            return (new_seq.tolist(),new_seq.tolist())
        else:
            return (seq,seq)

    def border_padding(self,seq,seed, max_len,shuffle):
        """For sequences shorter than seq_max_len introduced padding in the beginning and the ends of the sequences.
        If the amount of padding needed is divisible by 2 then the padding is shared evenly at the bginning and the end of the sequence.
        Otherwise randomly, the beginning or the end of the sequence will receive more padding"""
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            half_pad = pad / 2
            even_pad = [True if pad % 2 == 0 else False][0]
            if even_pad:#same amount of padding added at the beginning and the end of the sequence
                idx_pads = np.concatenate(
                    [np.arange(0, int(half_pad)), np.arange(max_len - int(half_pad), max_len)])
            else:
                idx_choice = np.array(random.sample(range(0, 1), 1),dtype=int).item()  # random choice of adding the extra padding to the beginning or end
                idx_pads_dict = {0: np.concatenate([np.arange(0, int(half_pad) + 1), np.arange(max_len - int(half_pad), max_len)]),
                                 1: np.concatenate([np.arange(0, int(half_pad)),np.arange(max_len - (int(half_pad) + 1), max_len)])}
                idx_pads = idx_pads_dict[idx_choice]

            new_seq = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx_pads] = False
            new_seq[mask] = np.array(seq)
            return (new_seq.tolist(),new_seq.tolist())
        else:
            return (seq,seq)

    def replicated_border_padding(self, seq,seed, max_len,shuffle):
        """
        Inspired by "replicated" padding in Convolutional NN https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        For sequences shorter than seq_max_len introduced padding in the beginning and the ends of the sequences.
        If the amount of padding needed is divisible by 2 then the padding is shared evenly at the bginning and the end of the sequence.
        Otherwise randomly, the beginning or the end of the sequence will receive more padding"""
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        #random.seed(91)
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            half_pad = pad / 2
            even_pad = [True if pad % 2 == 0 else False][0]
            if even_pad:  # same amount of paddng added at the beginning and the end of the sequence
                start = np.arange(0, int(half_pad))
                end = np.arange(max_len - int(half_pad), max_len)
                idx_pads = np.concatenate(
                    [start, end])
            else:
                idx_choice = np.array(random.sample(range(0, 1), 1),
                                      dtype=int).item()  # random choice of adding the extra padding to the beginning or end
                start_0 = np.arange(0, int(half_pad) + 1)
                end_0 = np.arange(max_len - int(half_pad), max_len)
                start_1 = np.arange(0, int(half_pad))
                end_1 = np.arange(max_len - (int(half_pad) + 1), max_len)
                idx_pads_dict = {
                    0: [np.concatenate([start_0, end_0]),start_0,end_0],
                    1: [np.concatenate([start_1, end_1]),start_1,end_0]}
                idx_pads,start,end = idx_pads_dict[idx_choice]

            new_seq = np.array(["#"] * max_len)
            new_seq_mask = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx_pads] = False
            new_seq[mask] = np.array(seq)
            new_seq_mask[mask] = np.array(seq)
            if start.size != 0 and end.size == 0:
                new_seq[~mask] = np.array(seq[:len(start)])
            elif start.size == 0 and end.size != 0:
                    new_seq[~mask] = np.array(seq[-len(end):])
            else:
                new_seq[~mask] = np.concatenate([np.array(seq[:len(start)]),np.array(seq[-len(end):])])
            return (new_seq.tolist(), new_seq_mask.tolist())
        else:
            return (seq,seq)

class ArtificialEpitopes(object):
    """Performs padding of a list of given sequences to a given len"""
    def __init__(self,sequences,allele_anchors,method,aa_dict):
        self.sequences = sequences
        self.allele_anchors = allele_anchors
        self.method = method
        self.aa_dict = aa_dict

    def weight_aminoacids(self,aa_dict):
        aa_groups_colors_dict,aa_groups_dict,groups_names_colors_dict,aa_by_groups_dict=VegvisirUtils.aminoacids_groups(aa_dict)

        weights_dict = defaultdict()
        for aa in aa_dict.keys():
            pass
        return aa_by_groups_dict

    def run(self):

        weighted_aminoacids_dict = self.weight_aminoacids(self.aa_dict)

        exit()

        artificial_sequences = {"artificial_1": list(map(lambda seq,anchors: self.artificial_1(seq,anchors),self.sequences,self.allele_anchors))}

        if self.method not in artificial_sequences.keys():
            raise ValueError("Padding method <{}> not implemented, please choose among <{}>".format(self.method,artificial_sequences.keys()))
        else:
            return artificial_sequences[self.method]

    def artificial_1(self,seq,anchors):


        new_aa = random.draw = choice(list_of_candidates, number_of_items_to_pick,p=probability_distribution)

        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        return (random.sample(seq, len(seq)),random.sample(seq, len(seq)))
































