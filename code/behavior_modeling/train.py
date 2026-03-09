import torch
import torch.optim as optim
import logging
import os
import torch.optim.lr_scheduler as LR_scheduler
import argparse
from pathlib import Path
import json
import numpy as np
from sklearn import linear_model as regression
from tqdm import tqdm
from read_data import load_fav_sequences
from model import ego_acc_LSTM_dist
from criterion import dist_loss


# Parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/MicroSimACC', help='Directory containing csv files')
    parser.add_argument('--data_csv', type=str, default=None, help='Single CSV file to use')
    parser.add_argument('--fav_id', type=int, default=0, help='Target FAV ID')
    parser.add_argument('--seq_len', type=int, default=1, help='Sequence length for LSTM input')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--result_dir', type=str, default=None, help='Output directory for model/checkpoints')
    
    return parser.parse_args()


def scaling_law(acc, K=20, a_min=0.1, a_max=1000.0):
    acc = np.array(acc)
    mean = np.mean(acc)
    std = np.std(acc)

    prob = []
    n = np.linspace(0.01, K, 300)
    for i in tqdm(n, desc="Calculating probability"):
        prob.append(np.sum((acc > mean + i * std) | (acc < mean - i * std)) / len(acc))
        if prob[-1] == 0.0:
            prob = prob[:-1]
            n = n[: len(prob)]
            break
    prob = np.array(prob)
    n = n[prob > 1 / len(acc) * 10]
    prob = prob[prob > 1 / len(acc) * 10]
    n = n[prob < 1.0]
    prob = prob[prob < 1.0]

    D = n * std
    x = np.log(prob)

    best_R2 = 0
    best_a = None
    best_k = None
    for a in np.linspace(a_min, a_max, int((a_max - a_min) * 10)):
        y_tmp = np.log(D / a + 1)
        reg = regression.LinearRegression(fit_intercept=False)
        reg.fit(x.reshape(-1, 1), y_tmp)
        k = reg.coef_[0]
        R2 = reg.score(x.reshape(-1, 1), y_tmp)
        if R2 > best_R2:
            best_R2 = R2
            best_k = k
            best_a = a

    return best_a, best_k, best_R2


def calibrate_distribution(model, data_loader, device, result_dir, logger):
    """Calibrate distribution parameters and fit scaling-law on validation set."""
    if data_loader is None or len(data_loader) == 0:
        logger.info('Calibration skipped: no validation data.')
        return

    model.eval()
    residuals = []
    sigmas = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)
            mu, sigma = model(X_batch)
            mu = mu.view(-1)
            sigma = sigma.view(-1)
            y = y_batch.view(-1)
            resid = (y - mu).detach().cpu()
            residuals.append(resid)
            sigmas.append(sigma.detach().cpu())

    if not residuals or not sigmas:
        logger.info('Calibration skipped: empty residuals.')
        return

    residuals = torch.cat(residuals)
    sigmas = torch.cat(sigmas)
    bias = residuals.mean().item()
    mae = residuals.abs().mean().item()
    sigma_mean = sigmas.mean().item()
    sigma_scale = mae / (sigma_mean + 1e-8)

    calib = {
        "bias": bias,
        "mae": mae,
        "sigma_mean": sigma_mean,
        "sigma_scale": sigma_scale,
    }

    residuals_np = residuals.detach().cpu().numpy()
    if np.std(residuals_np) > 1e-8:
        best_a, best_k, best_R2 = scaling_law(residuals_np)
        calib["a"] = float(best_a)
        calib["k"] = float(best_k)
        calib["scaling_law_R2"] = float(best_R2)
        logger.info(f"Scaling law fit: a={best_a}, k={best_k}, R2={best_R2}")
    else:
        logger.info("Scaling law fit skipped: residual std too small.")
    with open(os.path.join(result_dir, 'calibration.json'), 'w') as f:
        json.dump(calib, f, indent=2)
    logger.info(f'Calibration results saved: {calib}')




def main(model_type = 'LSTM'):
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #log file
    logger = logging.getLogger('')
    logger.handlers.clear()
    root = 'results'
    os.makedirs(root, exist_ok=True)
    if args.result_dir:
        result_file = args.result_dir
        os.makedirs(result_file, exist_ok=True)
    else:
        # new log file folder, e.g. results/MicroSimACC_FAV0_seq10
        tag = f'MicroSimACC_FAV{args.fav_id}_seq{args.seq_len}'
        result_file = os.path.join(root, tag)
        os.makedirs(result_file, exist_ok=True)
    # set the log file path
    filehandler = logging.FileHandler(os.path.join(result_file, 'training.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)


    # Set hyperparameters
    output_steps = 1
    batch_size = args.batch_size
    hidden_size = 128
    num_layers = 2
    num_epochs = args.num_epochs
    lr = 0.0001
    num_dir = 1
    
    logger.info('*'*50)
    logger.info(f'output_steps: {output_steps}, batch_size: {batch_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, num_epochs: {num_epochs}, lr: {lr}')
    logger.info('*'*50)
    logger.info(f'data_dir: {args.data_dir}, fav_id: {args.fav_id}, seq_len: {args.seq_len}, train_ratio: {args.train_ratio}')
    logger.info('*'*50)
    # Load dataset
    data_root = Path(args.data_dir)
    dataset = load_fav_sequences(
        data_dir=str(data_root),
        fav_id=args.fav_id,
        seq_len=args.seq_len,
        csv_file=args.data_csv,
    )
    total_len = len(dataset)
    if total_len == 0:
        logger.error(f'No data found for FAV {args.fav_id} in {data_root}.')
        return
    train_len = int(total_len * args.train_ratio)
    train_len = min(total_len - 1, train_len) if total_len > 1 else total_len
    train_len = max(1, train_len)
    test_len = total_len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_len > 0 else None

    logger.info('The number of sequences for training: {}'.format(len(train_dataset)))
    logger.info('The number of sequences for validation: {}'.format(len(test_dataset)))
    
    # Create model, loss function, and optimizer
    num_feature = dataset.tensors[0].shape[-1]
    if model_type == 'LSTM_dist':
        model = ego_acc_LSTM_dist(num_feature = num_feature, hidden_size = hidden_size, num_layers = num_layers, output_size = output_steps, NUMDIR=num_dir)
        criterion = dist_loss()
    else:
        raise NotImplementedError
    model = model.to(device=device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Set up the learning rate scheduler
    lr_scheduler = LR_scheduler.MultiStepLR(optimizer, milestones=[60,90], gamma=0.1)

    best_mae = 9999

    # Train model
    logger.info('############# Start Training #############')
    for epoch in range(num_epochs):
        # current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # logger.info(f'############# Starting Epoch {epoch+1} | LR: {current_lr} #############')
        
        train_loss = 0
        #train_loader = tqdm(train_loader, dynamic_ncols=True)
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device).view(-1, output_steps, num_dir)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # Update learning rate
        lr_scheduler.step()


        if (epoch+1) % 1 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.6f}')
        

            # Evaluation
            if test_loader is None or len(test_loader) == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], No validation set available.')
            else:
                test_MAE = 0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(device=device)
                        y_batch = y_batch.to(device=device).view(-1, output_steps, num_dir)
                        y_pred = model(X_batch)
                        
                        if model_type == 'LSTM_dist':
                            y_pred = y_pred[0] # Mean value
                        
                        loss = torch.mean(torch.abs(y_pred - y_batch))
                        test_MAE += loss.item()

                MAE = test_MAE/len(test_loader)
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], MAE (Acc_FAV): {MAE:.4f}')
                
                if MAE < best_mae:
                    best_mae = MAE
                    # save the best model
                    torch.save(model.state_dict(), f'{result_file}/best.pth')

    # Calibration on validation set using best model (if available)
    if test_loader is not None and len(test_loader) > 0:
        best_path = os.path.join(result_file, 'best.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        calibrate_distribution(model, test_loader, device, result_file, logger)

    # save the final model
    torch.save(model.state_dict(), f'{result_file}/final.pth')



if __name__ == '__main__':
    main(model_type='LSTM_dist')
