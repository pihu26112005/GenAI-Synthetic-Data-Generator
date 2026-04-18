import numpy as np
import os

import src
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# class TabularDatasetCustom(Dataset):
#     def __init__(self, X_num, X_cat, y):
#         self.X_num = X_num
#         self.X_cat = X_cat
#         self.y = y

#     def __getitem__(self, index):
#         this_num = self.X_num[index]
#         this_cat = self.X_cat[index]
#         this_y = self.y[index]

#         sample = (this_num, this_cat, this_y)

#         return sample

#     def __len__(self):
#         return self.X_num.shape[0]
        
# class TabularDataset(Dataset):
#     def __init__(self, X_num, X_cat):
#         self.X_num = X_num
#         self.X_cat = X_cat

#     def __getitem__(self, index):
#         this_num = self.X_num[index]
#         this_cat = self.X_cat[index]

#         sample = (this_num, this_cat)

#         return sample

#     def __len__(self):
#         return self.X_num.shape[0]

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]
        this_y = self.y[index]
        
        # Yield the tuple directly; main.py will batch them
        return this_num, this_cat, this_y

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
        
        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)


def get_tbs_sampler(y_train, lambda_tbs=0.5):
    """
    Creates a WeightedRandomSampler based on CTTVAE's Training-by-Sampling (TBS) PMF.
    Interpolates between the original class distribution and a uniform distribution.
    """
    # Ensure y_train is a flattened integer array
    y_train_int = np.array(y_train).astype(int).squeeze()
    
    # Calculate class distributions
    class_counts = np.bincount(y_train_int)
    total_samples = len(y_train_int)
    
    P_orig = class_counts / total_samples
    
    # For ternary ORD labels, len(class_counts) should be 3
    P_uniform = np.ones(len(class_counts)) / len(class_counts)
    
    # Phase 3: The PMF Equation
    # PMF[y] = \lambda * P_orig[y] + (1-\lambda) * (1/3)
    PMF = lambda_tbs * P_orig + (1 - lambda_tbs) * P_uniform
    
    # The weight applied to an instance is proportional to Target PMF / Original P
    class_weights = PMF / P_orig
    
    # Map the class weight to each individual sample in the dataset
    sample_weights = torch.tensor([class_weights[y] for y in y_train_int], dtype=torch.float)
    
    # Instantiate the PyTorch sampler
    tbs_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )
    
    return tbs_sampler