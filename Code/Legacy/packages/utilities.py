"""Utilities functions. Stuff than can be used in general for parallel tempering and often comes in handy"""

#imports

#general imports
import numpy as np
import torch

#C_v definition and functions that generate temp_range

def Cv(T, N):
    """Approximation of heat capacity C_V by FRT.

    Parameters:
    - T: temperature,
    - N: number of spins.

    Returns:
    - C_V: heat capacity"""
    g = 0.05 * T + 2.0 * T * T * T
    return N * g / (1.0 + (g * 0.5 * T * T)**0.4)**2.5

def Cv_3d(T, N):
    """Approximation of heat capacity C_V by FRT for 3D Ising model.

    Parameters:
    - T: temperature,
    - N: number of spins.
    
    Returns:
    - C_V: heat capacity"""
    return N * 34.19 * T * T / (10.36 + T * T * T) / (10.36 + T * T * T)

def get_temps(start, finish, N):
    """
    Generate temperatures using the heat capacity policy.

    Parameters:
    - start (float): Starting temperature.
    - finish (float): Target temperature to stop the generation.
    - N (int): Number of spins

    Returns:
    - numpy.ndarray: Array of generated temperatures.

    The function generates a sequence of temperatures starting from 'start' and
    continuing until reaching or exceeding 'finish' using a specific policy. The
    policy involves modifying the temperature 'T' based on the optimal thermal step of the Parallel
    Tempering procedure.
    """
    T = start  # Initialize the temperature with the starting value
    temperatures = [T]  # Create a list to store generated temperatures, starting with the initial temperature

    # Continue generating temperatures until the temperature reaches or exceeds the 'finish' value
    while T < finish:
        # Update the temperature using a specific formula (commented out line can be an alternative)
        T /= (1.0 - 1.684 / np.sqrt(Cv(T, N)))
        # Uncomment the line below if the approximate version is used
        # T *= 1.0 + 1.684 / np.sqrt(Cv(T, N))

        temperatures.append(T)  # Append the updated temperature to the list

    return np.array(temperatures)  # Convert the list to a numpy array and return it

def get_betas(start, finish, factor, N):
    """
    Generate beta values using a specific factor as a multiplying factor of 1/sqrt(Cv).

    Parameters:
    - start: Starting beta value.
    - finish: Target beta value to stop the generation.
    - factor: Multiplying factor for 1/sqrt(Cv) in the generation process.
    - N: Number of spins
    Returns:
    -betas: Array of generated beta values.

    The function generates a sequence of beta values starting from 'start' and
    continuing until reaching or exceeding 'finish'. The generation process involves
    modifying beta based on the PT formula. The
    generated beta values are stored in a numpy array and returned.
    """

    beta = start  # Initialize the beta value with the starting value
    betas = [beta]  # Create a list to store generated beta values, starting with the initial beta

    # Continue generating beta values until the beta value reaches or exceeds the 'finish' value
    while beta < finish:
        # Update beta using a specific formula
        beta += factor / np.sqrt(Cv(1 / beta, N))

        betas.append(beta)  # Append the updated beta value to the list

    return np.array(betas)  # Convert the list to a numpy array and return it

def get_betas_3d(start, finish, factor, N):
    """
    Generate beta values using a specific factor as a multiplying factor of 1/sqrt(Cv) for Ising 3d.

    Parameters:
    - start: Starting beta value.
    - finish: Target beta value to stop the generation.
    - factor: Multiplying factor for 1/sqrt(Cv) in the generation process.
    - N: Number of spins
    Returns:
    -betas: Array of generated beta values.

    The function generates a sequence of beta values starting from 'start' and
    continuing until reaching or exceeding 'finish'. The generation process involves
    modifying beta based on the PT formula. The
    generated beta values are stored in a numpy array and returned.
    """

    beta = start  # Initialize the beta value with the starting value
    betas = [beta]  # Create a list to store generated beta values, starting with the initial beta

    # Continue generating beta values until the beta value reaches or exceeds the 'finish' value
    while beta < finish:
        # Update beta using a specific formula
        beta += factor / np.sqrt(Cv_3d(1 / beta, N))

        betas.append(beta)  # Append the updated beta value to the list

    return np.array(betas)  # Convert the list to a numpy array and return it

def compute_energy(s, J, take_mean=False):
    """
    Compute the energy for each configuration using the Edwards-Anderson (EA) energy formula.

    Parameters:
    - s (torch.Tensor): Tensor representing the spin configurations. It is a #configurations x #numberofspin tensor.
    - J (torch.Tensor): Tensor representing the interaction matrix. It is a #numberofspin x #numberofspin tensor.
    - take_mean (bool): If True, compute the mean energy across all configurations. If False, return energies for each configuration.

    Returns:
    - torch.Tensor: Tensor containing the computed energy for each configuration or the mean energy.

    The function calculates the EA energy for each configuration in tensor 's' using
    the interaction matrix 'J'. It involves tensor contractions and multiplication
    operations. The computed energy values are returned as a tensor. If 'take_mean' is
    True, it returns the mean energy across all configurations.
    """
    # Perform tensor operations to calculate the EA energy for each configuration
    energy = -(torch.einsum("ki,ik->k", s, torch.einsum("ij,kj->ik", J, s))) / 2
    if not take_mean:
        return energy  # Return the computed energy tensor for each configuration
    else:
        return energy.mean()  # Return the mean energy across all configurations
    
def hamming_distance(data, GS_config):
    # Calculate the absolute differences between GS_config and each row in data
    distance1 = torch.sum(torch.abs(GS_config - data), axis=1)
    
    # Calculate the absolute differences between the negation of GS_config and each row in data
    distance2 = torch.sum(torch.abs(GS_config + data), axis=1)
    
    # Take the element-wise minimum of distance1 and distance2
    x = torch.min(distance1, distance2)
    
    # Calculate the mean of the minimum distances and divide by 2
    result = torch.mean(x) / 2
    
    return result

def corr(x, tau):
  """Compute autocorrelation function"""
  if tau == 0:
    return torch.mean(x*x)
  else:
    return torch.mean(x[tau:,:]*x[:-tau, :])-torch.mean(x[tau:,:])*torch.mean(x[:-tau,:])
