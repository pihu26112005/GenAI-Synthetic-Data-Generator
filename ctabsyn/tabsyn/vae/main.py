import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset, get_tbs_sampler

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


# def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
#     ce_loss_fn = nn.CrossEntropyLoss()
#     mse_loss = (X_num - Recon_X_num).pow(2).mean()
#     ce_loss = 0
#     acc = 0
#     total_num = 0

#     for idx, x_cat in enumerate(Recon_X_cat):
#         if x_cat is not None:
#             ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
#             x_hat = x_cat.argmax(dim = -1)
#         acc += (x_hat == X_cat[:,idx]).float().sum()
#         total_num += x_hat.shape[0]
    
#     ce_loss /= (idx + 1)
#     acc /= total_num
#     # loss = mse_loss + ce_loss

#     temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

#     loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
#     return mse_loss, ce_loss, loss_kld, acc

def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat):
    # Notice we don't even need to pass mu_z or logvar_z here anymore!
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    if total_num > 0:
        ce_loss /= (idx + 1)
        acc /= total_num
    else:
        acc = torch.tensor(0.0).to(X_num.device)

    # KLD calculation is entirely removed.
    return mse_loss, ce_loss, acc

def mmd_loss(z_sampled, z_prior):
    """
    Computes the Maximum Mean Discrepancy (MMD) between the sampled latent vectors 
    and a standard Gaussian prior using an RBF kernel.
    """
    def rbf_kernel(x, y, sigma=1.0):
        beta = 1.0 / (2.0 * sigma**2)
        dist = torch.cdist(x, y)**2
        return torch.exp(-beta * dist)
        
    k_xx = rbf_kernel(z_sampled, z_sampled)
    k_yy = rbf_kernel(z_prior, z_prior)
    k_xy = rbf_kernel(z_sampled, z_prior)
    
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

def ordinal_triplet_loss(mu, labels, m_close=1.0, m_far=2.5):
    """
    Computes the Ordinal Triplet Loss (Fully Vectorized for GPU).
    """
    assert m_far >= 2 * m_close, f"Crucial Constraint violated: m_far ({m_far}) must be >= 2 * m_close ({2 * m_close})"

    # Calculate distance matrix for the entire batch at once [Batch, Batch]
    dist_matrix = torch.cdist(mu, mu, p=2) ** 2
    labels = labels.squeeze()
    B = labels.size(0)

    # Create 2D grids of our labels to compare everything simultaneously
    labels_col = labels.unsqueeze(1) # Shape: [B, 1]
    labels_row = labels.unsqueeze(0) # Shape: [1, B]

    # Mask for Positives: same label, but completely ignore the diagonal (self-distance)
    pos_mask = (labels_col == labels_row) & ~torch.eye(B, dtype=torch.bool, device=mu.device)
    
    # Mask for Negatives: different label
    neg_mask = (labels_col != labels_row)

    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    # 1. Find Hardest Positives (d_ap)
    # Fill non-positives with -1.0 so they are ignored by the max() function
    masked_pos_dists = dist_matrix.masked_fill(~pos_mask, -1.0)
    d_ap = masked_pos_dists.max(dim=1)[0] # Shape: [B]
    
    # 2. Determine Target Margins based on Ordinal Distance
    class_dist = torch.abs(labels_col - labels_row)
    m_target = torch.where(class_dist == 1, m_close, torch.where(class_dist == 2, m_far, 0.0))

    # 3. Calculate Loss Matrix for all possibilities
    d_ap_expanded = d_ap.unsqueeze(1) # Shape: [B, 1]
    loss_matrix = torch.clamp(d_ap_expanded - dist_matrix + m_target, min=0.0)

    # 4. Filter for Semi-Hard Negatives (d_ap < d_an < d_ap + m_target)
    semi_hard_mask = neg_mask & (dist_matrix > d_ap_expanded) & (dist_matrix < d_ap_expanded + m_target)

    # Extract only the losses that meet the semi-hard criteria
    valid_losses = loss_matrix[semi_hard_mask]

    # 5. Average the loss, with a fallback if no semi-hard negatives exist
    if valid_losses.numel() > 0:
        return valid_losses.mean()
    else:
        # Fallback: just use the hardest negative (closest different class)
        masked_neg_dists = dist_matrix.masked_fill(~neg_mask, float('inf'))
        d_an, neg_indices = masked_neg_dists.min(dim=1) # Shape: [B]
        
        hardest_class_dist = torch.abs(labels - labels[neg_indices])
        m_target_fallback = torch.where(hardest_class_dist == 1, m_close, m_far)
        
        fallback_losses = torch.clamp(d_ap - d_an + m_target_fallback, min=0.0)
        valid_rows = neg_mask.any(dim=1)
        
        return fallback_losses[valid_rows].mean() if valid_rows.any() else torch.tensor(0.0, device=mu.device)

def main(args):
    dataname = args.dataname
    data_dir = f'data/{dataname}'

    # max_beta = args.max_beta
    # min_beta = args.min_beta
    # lambd = args.lambd
    beta = args.beta
    alpha = args.alpha

    device =  args.device


    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    # X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)


    # train_data = TabularDataset(X_train_num.float(), X_train_cat)

    # 1. Load the labels!
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_test = np.load(f'{data_dir}/y_test.npy') # If you have a test set for labels

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)
    y_train, y_test = torch.tensor(y_train).long(), torch.tensor(y_test).long()

    # 2. Pass labels to the dataset
    train_data = TabularDataset(X_train_num.float(), X_train_cat, y_train)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)
    y_test = y_test.to(device)

    # 3. Create the TBS Sampler
    tbs_sampler = get_tbs_sampler(y_train.numpy(), lambda_tbs=0.5)

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        # shuffle = True,
        sampler = tbs_sampler,
        num_workers = 4,
    )

    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
    model = model.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

    num_epochs = 4000
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0
    max_patience = 150 # For early stopping

    # beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        # curr_loss_kl = 0.0

        # Add tracking variables for the new losses
        curr_loss_mmd = 0.0
        curr_loss_triplet = 0.0

        curr_count = 0

        for batch_num, batch_cat, batch_y in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)

            # kld term removed
            loss_mse, loss_ce, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat)

            # loss = loss_mse + loss_ce + beta * loss_kld

            # ====== THE FIX: FLATTEN THE LATENT TENSORS ======
            # mu_z is shape [batch_size, num_columns, d_token]
            # We flatten it to [batch_size, num_columns * d_token]
            batch_size = mu_z.shape[0]
            flat_mu_z = mu_z.view(batch_size, -1)
            flat_std_z = std_z.view(batch_size, -1)
            # =================================================

            # Sample z using the reparameterization trick (already happening inside model, but we need z for MMD)
            eps = torch.randn_like(flat_std_z)
            z_sampled = flat_mu_z + eps * torch.exp(0.5 * flat_std_z)

            # Sample prior from standard normal
            z_prior = torch.randn_like(z_sampled)
            
            # Calculate MMD
            loss_mmd = mmd_loss(z_sampled, z_prior)
            
            # Calculate Ordinal Triplet Loss
            loss_triplet = ordinal_triplet_loss(flat_mu_z, batch_y, m_close=1.0, m_far=2.5)

            # The New Hybrid Objective
            alpha = 1.0 # Triplet weight
            loss = loss_mse + loss_ce + beta * loss_mmd + alpha * loss_triplet

            ##### add loss here
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            # curr_loss_kl    += loss_kld.item() * batch_length

            # Update tracking metrics
            curr_loss_mmd += loss_mmd.item() * batch_length
            curr_loss_triplet += loss_triplet.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        # kl_loss = curr_loss_kl / curr_count
        mmd_loss_val = curr_loss_mmd / curr_count
        triplet_loss_val = curr_loss_triplet / curr_count
        

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            # kld term removed
            val_mse_loss, val_ce_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat)
            # val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()   

            # ====== THE FIX: FLATTEN THE LATENT TENSORS ======
            # mu_z is shape [batch_size, num_columns, d_token]
            # We flatten it to [batch_size, num_columns * d_token]
            batch_size = mu_z.shape[0]
            flat_mu_z = mu_z.view(batch_size, -1)
            flat_std_z = std_z.view(batch_size, -1)
            # =================================================

            # 1. Calculate Validation MMD
            eps = torch.randn_like(flat_std_z)
            val_z_sampled = flat_mu_z + eps * torch.exp(0.5 * flat_std_z)
            val_z_prior = torch.randn_like(val_z_sampled)
            val_mmd_loss = mmd_loss(val_z_sampled, val_z_prior)
            
            # 2. Calculate Validation Triplet Loss
            val_triplet_loss = ordinal_triplet_loss(flat_mu_z, y_test, m_close=1.0, m_far=2.5)

            # 3. Validation Objective
            val_loss = val_mse_loss.item() + val_ce_loss.item() + beta * val_mmd_loss.item() + alpha * val_triplet_loss.item() 

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                #Adding early stopping

                # Early Stopping Logic
                if patience >= max_patience:
                    print(f"\nEarly stopping triggered! Validation loss hasn't improved for {max_patience} epochs.")
                    break # This exits the epoch loop immediately

                # if patience == 10:
                #     if beta > min_beta:
                #         beta = beta * lambd


        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train MMD:{:.6f}, Train Triplet:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, mmd_loss_val, triplet_loss_val, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))
        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))
        print('epoch: {}, Train MSE: {:.6f}, Train CE:{:.6f}, Train MMD:{:.6f}, Train Triplet:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Val MMD:{:.6f}, Val Triplet:{:.6f}'.format(
            epoch, num_loss, cat_loss, mmd_loss_val, triplet_loss_val, 
            val_mse_loss.item(), val_ce_loss.item(), val_mmd_loss.item(), val_triplet_loss.item()
        ))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    # parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    # parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    # parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')
    parser.add_argument('--beta', type=float, default=1.0, help='Static weight for the MMD loss.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Static weight for the Ordinal Triplet loss.')

    args = parser.parse_args()
    
    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    # ADD THIS EXACT LINE TO ACTUALLY START THE TRAINING
    main(args)
