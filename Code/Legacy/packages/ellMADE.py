#imports

#general imports
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import random

#torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class WeightClipper(object):
    """Autoregressive contstraint"""
    def __init__(self, param_mask, frequency=1):
        self.frequency = frequency
        self.param_mask = param_mask

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w *= self.param_mask
            module.weight.data = w

# Build the autoregressive model
class ellmade(nn.Module):
    """Autoregressive Made"""
    def __init__(self, input_size):
        super(ellmade, self).__init__()
        self.layer = nn.Linear(input_size, input_size, bias=False)
        #self.constraint = AutoregressiveConstraint()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        #x = self.constraint(x) #non sono sicuro funzioni
        x = self.activation(2*x)
        torch.cuda.empty_cache()
        return x

# Train the model
def train_ellmade(data, input_size, param_mask, epochs=40, batch_size=256, learning_rate=1e-3):
    """Train the made architecture using data"""
    model=ellmade(input_size)
    model = model.to("cuda")
    model.train()
    clipper = WeightClipper(param_mask)
    model.apply(clipper)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction = "sum")

    for epoch in tqdm(range(epochs)):
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            batch_data = data[indices].to("cuda")
            #batch_data = Variable(torch.FloatTensor(batch_data))

            # Forward pass
            output = model(batch_data)

            # Compute loss
            loss = criterion(output, (batch_data + 1) / 2)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.apply(clipper)

    return model

def generate_config_ellmade(model, N_spins, N_config):
    """Generate N_config of N_spins using the autoregressive model"""
    with torch.no_grad():
        config = (torch.bernoulli(torch.full((N_config, N_spins), 0.5))*2-1).to("cuda")
        for n in range(N_spins):
            probs = model(config)
            config[:, n] =  (torch.bernoulli(probs[:,n])*2-1)
            torch.cuda.empty_cache()
        return config
    
#the mask for the parameters, to set them to 0 when needed

def get_param_mask(N, layers, J):
  """
  Generates a parameter mask based on the interaction matrix J and the number of layers.

  The mask is constructed by iteratively multiplying the interaction matrix
  (with an added identity matrix) by itself for the specified number of layers.
  This effectively captures the indirect interactions between spins up to the given depth.

  Parameters:
  - N: number of spins.
  - layers: The number of layers to consider for indirect interactions.
  - J: matrix of couplings.

  Returns:
  - param_mask: A binary mask of the same shape as J, where 1 indicates a parameter to be learned and 0 indicates the absence of a parameter.
  """

  J_eye = J + torch.eye(N)  # Add identity matrix to account for self-interactions

  matrix = J_eye
  for i in range(layers):
    matrix = torch.matmul(J_eye, matrix)  # Iteratively multiply to capture indirect interactions

  param_mask = torch.zeros((N,N))  # Initialize the mask to all zeros
  param_mask[matrix != 0] = 1
  # Set elements corresponding to nonzero interactions to 1

  return param_mask

def masker(N, layer, J):
    M = get_param_mask(N, layer, J)
    M = M*torch.tril(torch.ones((N, N)), -1)
    return M