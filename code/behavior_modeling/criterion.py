import torch
import torch.nn as nn
import numpy as np


# distribution loss
class dist_loss(nn.Module):
    def __init__(self):
        super(dist_loss, self).__init__()
    
    def forward(self, outputs, y):
        mu, sigma = outputs
        l_mean = (mu - y) ** 2 # (B, output_steps, 2) for highD
        l_std = (torch.abs(mu - y) - sigma)**2
        loss = l_mean + l_std
        
        return torch.mean(loss)
    

# Function to calculate the R^2 value for the goodness of fit
def calculate_r2(y_observed, y_predicted):
    ss_res = np.sum((y_observed - y_predicted)**2)       # Residual sum of squares
    ss_tot = np.sum((y_observed - np.mean(y_observed))**2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)


# Function to calculate the KL divergence between two distributions
# p: empirical distribution, q: fitted distribution
def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    # Avoid division by zero and log of zero
    p = p + 1e-10
    q = q + 1e-10
    
    # Normalize to ensure they are proper probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))


# Function to calculate log-likelihood
def log_likelihood(y_predicted):
    y_predicted = np.asarray(y_predicted, dtype=np.float32)

    # Avoid log of zero
    y_predicted = y_predicted + 1e-10

    # Normalize to ensure it is a proper probability distribution
    y_predicted = y_predicted / np.sum(y_predicted)

    return np.sum(np.log(y_predicted))