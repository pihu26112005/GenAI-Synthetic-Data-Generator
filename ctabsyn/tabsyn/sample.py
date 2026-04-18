import torch

import argparse
import warnings
import time

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    # cond_val = args.condition_by
    # n_classes = args.n_classes

    # #####
    # # if cond_val == 0 , one_hot = 1,0,0
    # # if cond_val == 1 , one_hot = 0,1,0
    # # if cond_val == 2 , one_hot = 0,0,1
    # one_hot = torch.zeros(n_classes)
    # one_hot[cond_val] = 1
 
    # label = torch.tensor(one_hot)
    # label_dim = label.shape[0]
    # label = label.to(device)
    # #####

    # train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    # in_dim = train_z.shape[1] 

    # mean = train_z.mean(0)
    # ######

    # REMOVE cond_val and one_hot logic entirely. 
    # Replace with this explicit batch creation:
    
    n_class_0 = args.n_class_0
    n_class_2 = args.n_class_2
    num_samples = n_class_0 + n_class_2
    
    if num_samples == 0:
        raise ValueError("You must specify a number of samples to generate using --n_class_0 and/or --n_class_2")

    # Create a batched tensor of integer labels (e.g., [0,0,0..., 2,2,2...])
    labels_0 = torch.zeros(n_class_0, dtype=torch.long)
    labels_2 = torch.full((n_class_2,), 2, dtype=torch.long)
    
    label_tensor = torch.cat([labels_0, labels_2]).to(device)

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)


    # denoise_fn = MLPDiffusion(in_dim, label_dim, 1024).to(device)
    # Explicitly set n_classes=3 for the ternary ORD system
    denoise_fn = MLPDiffusion(in_dim, n_classes=3, dim_t=1024).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    # num_samples = train_z.shape[0]
    # sample_dim = in_dim

    # #####
    # x_next = sample(model.denoise_fn_D, num_samples, sample_dim, label)
    
    sample_dim = in_dim

    #####
    # Pass our custom label_tensor containing the exact 0s and 2s requested
    x_next = sample(model.denoise_fn_D, num_samples, sample_dim, label_tensor)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    # add arguments
    parser.add_argument('--n_class_0', type=int, default=0, help='Number of majority class samples to generate')
    parser.add_argument('--n_class_2', type=int, default=0, help='Number of minority class samples to generate')
    parser.add_argument('--save_path', type=str, default='synthetic_data.csv', help='Path to save the generated data')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'