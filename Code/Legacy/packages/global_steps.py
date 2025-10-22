"""Definitions necessary for the global steps"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random


def generate_config(model, N_spins, N_config, J):
    """
    Generate N_config spin configurations with N_spins using the MADE.

    Parameters:
    - model: Autoregressive model used for spin generation.
    - N_spins: Number of spins in each configuration.
    - N_config: Number of configurations to generate.

    Returns:
    - Tensor: A tensor containing generated spin configurations with shape (N_config, N_spins).
    """

    with torch.no_grad():
        # Initialize a tensor with random binary configurations (values of -1 or 1)
        #config = (torch.bernoulli(torch.full((N_config, N_spins), 0.5)) * 2 - 1).to("cuda")
        #config = torch.ones((N_config, N_spins)).to("cuda")
        config = torch.zeros((N_config, N_spins)).to("cuda")
    
        # Generate each spin in an autoregressive manner
        for n in range(N_spins):
            # Get probabilities from the autoregressive model for the nth spin
            probs = model(config)
            
            # Sample new spin values based on probabilities and update the configuration
            config[:, n] = (torch.bernoulli(probs[:, n]) * 2 - 1)
            torch.cuda.empty_cache()
    
    return config

def generate_config_fast(model, N_spins, N_config, J):
    """
    Generate N_config spin configurations with N_spins using the MADE.

    Parameters:
    - model: Autoregressive model used for spin generation.
    - N_spins: Number of spins in each configuration.
    - N_config: Number of configurations to generate.

    Returns:
    - Tensor: A tensor containing generated spin configurations with shape (N_config, N_spins).
    """

    with torch.no_grad():
        # Initialize a tensor with random binary configurations (values of -1 or 1)
        #config = (torch.bernoulli(torch.full((N_config, N_spins), 0.5)) * 2 - 1).to("cuda")
        #config = torch.ones((N_config, N_spins)).to("cuda")
        config = torch.zeros((N_config, N_spins)).to("cuda")
    
        # Generate each spin in an autoregressive manner
        for n in range(N_spins):
            # Get probabilities from the autoregressive model for the nth spin
            probs = model.forward_n(config, n)
            
            # Sample new spin values based on probabilities and update the configuration
            config[:, n] = (torch.bernoulli(probs) * 2 - 1)
            torch.cuda.empty_cache()
    
    return config


def generate(model, current_config, new_beta, N, J, N_data = 100000, N_configs = 10):
    """
    Perform Metropolis-Hastings updates on spin configurations using the autoregressive model.

    Parameters:
    - model: Autoregressive model used for spin generation.
    - current_config: Tensor representing the current spin configuration.
    - new_beta: Inverse temperature for the Metropolis-Hastings update.
    - N: Number of spins.
    - J: Interaction matrix.
    - N_data: Number of data points to generate in each iteration (default is 100000).
    - N_configs: Number of configurations to generate in each iteration (default is 10).

    Returns:
    - Tuple: A tuple containing two elements.
        - Tensor: Updated spin configurations after Metropolis-Hastings updates.
        - float: Acceptance rate of proposed configurations.
    """

    with torch.no_grad():
    # Initialize acceptance rate counter
        acc_rate = 0
        
        # Binary Cross Entropy Loss function
        bce = nn.BCELoss(reduction="none")
        
        # Store the current configurations
        old_configs = torch.clone(current_config)
        torch.cuda.empty_cache()
        
        # Perform Metropolis-Hastings updates
        for t in tqdm(range(N_configs)):
            # Generate new configurations
            new_configs = generate_config(model, N, N_data, J)
            
            # Calculate energies and arguments for old and new configurations
            energy_old = -torch.einsum("ki,ik->k", old_configs, torch.einsum("ij,kj->ik", J, old_configs)) / 2
            arg_old = -new_beta * energy_old + torch.sum(bce(model(old_configs), (old_configs + 1) / 2), axis=1)
            
            new_energies = -torch.einsum("li,il->l", new_configs, torch.einsum("ij,lj->il", J, new_configs)) / 2
            arg_new = -new_beta * new_energies + torch.sum(bce(model(new_configs), (new_configs + 1) / 2), axis=1)
            
            # Acceptance probability calculation
            acc = (torch.log(torch.rand(size=(N_data,))).to("cuda") < (arg_new - arg_old)).int()
            #acc = (torch.log(torch.rand(size=(N_data,))).to("cuda") < (arg_new - arg_old))
            acc_rate += torch.sum(acc)
            
            # Update configurations based on acceptance
            old_configs = torch.einsum("i, ij->ij", (1 - acc), old_configs) + torch.einsum("i, ij->ij", acc, new_configs)
            #old_configs[acc] = new_configs[acc]
            
            torch.cuda.empty_cache()
        
        # Calculate and return the acceptance rate
        return old_configs, float(acc_rate / N_data / N_configs)