"""Made Class"""

#imports
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ExponentialLR

#set device to "cuda" if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoregressiveMasking(object):
    """Autoregressive constraint for weight matrices."""
    def __init__(self, frequency=1):
        """
        Constructor for AutoregressiveMasking.

        Parameters:
        - frequency (int): Controls how often the autoregressive constraint is applied.
        """
        self.frequency = frequency

    def __call__(self, module):
        """
        Applies the autoregressive constraint to the weight matrices of a module.

        Parameters:
        - module: PyTorch module to which the constraint is applied.
        """
        # Apply autoregressive constraint to weight matrices
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = torch.tril(w, -1)  # Apply lower triangular masking
            module.weight.data = w

# Build the autoregressive model
# Build the autoregressive model
class made(nn.Module):
    """Autoregressive MADE (Masked Autoencoder for Distribution Estimation)."""
    def __init__(self, input_size):
        """
        Constructor for the MADE model.

        Parameters:
        - input_size (int): Size of the input features.
        """
        super(made, self).__init__()
        self.layer = nn.Linear(input_size, input_size, bias=False)
        # self.constraint = AutoregressiveConstraint()  # Commented out, not used in forward pass
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the MADE model.

        Parameters:
        - x: Input tensor to the model.

        Returns:
        - x: Output tensor after the forward pass.
        """
        x = self.layer(x)
        # x = self.constraint(x)  # Commented out, not used in the forward pass
        x = self.activation(2 * x)  # Apply activation function
        if device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU cache
        return x
    
    def forward_n(self, input, n):
        # Get the n-th row of the weight matrix of the linear layer
        nth_row = self.layer.weight[n]
        x = torch.einsum("ij, j->i", input[:, :n], nth_row[:n])
        x = self.activation(2 * x)
        return x

# Train the model
def train_made(dataset, input_size, epochs=50, batch_size=256, learning_rate=1e-3):
    """
    Train the MADE architecture using data.

    Parameters:
    - data: Training data as a PyTorch tensor, aka #configurations x #spins tensor.
    - input_size (int): Size of the input features.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.

    Returns:
    - model: Trained MADE model.
    """
    data = torch.clone(dataset)
    model = made(input_size)
    model = model.to(device)
    model.train()
    clipper = AutoregressiveMasking()
    model.apply(clipper)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="sum")

    for epoch in tqdm(range(epochs)):
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            #batch_data = data[indices]
            batch_data = data[indices].to(device)

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

def train_made_improved(dataset, input_size, epochs=50, batch_size=256, patience = 10, learning_rate = 0.001, scheduler_time = 10):
    """
    Train the MADE architecture using data.

    Parameters:
    - data: Training data as a PyTorch tensor, aka #configurations x #spins tensor.
    - input_size (int): Size of the input features.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.

    Returns:
    - model: Trained MADE model.
    """
    data = torch.clone(dataset)
    model = made(input_size)
    model = model.to(device)
    model.train()
    clipper = AutoregressiveMasking()
    model.apply(clipper)

    best_loss = 100000000

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="sum")
    scheduler = ExponentialLR(optimizer, gamma=0.5)

    for epoch in range(epochs):
        tot_loss = 0
        count = 0
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            #batch_data = data[indices]
            batch_data = data[indices].to(device)

            # Forward pass
            output = model(batch_data)

            # Compute loss
            loss = criterion(output, (batch_data + 1) / 2)
            tot_loss += loss
            count += 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.apply(clipper)

        if epoch % scheduler_time == 1 and epoch > 0:
                scheduler.step()

        if tot_loss/count < best_loss:
            best_loss = tot_loss/count
            epochs_since_best_val_acc = 0
            best_weights = model.state_dict()
        else:
            epochs_since_best_val_acc += 1

        if epochs_since_best_val_acc >= patience:
            break
    model.load_state_dict(best_weights)
    return model

#In retrain, we do not put decay nor patience
def retrain_made(model, dataset, epochs=50, batch_size=256, learning_rate = 0.001):
    """
    Train the MADE architecture using data.

    Parameters:
    - data: Training data as a PyTorch tensor, aka #configurations x #spins tensor.
    - input_size (int): Size of the input features.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.

    Returns:
    - model: Trained MADE model.
    """
    data = torch.clone(dataset)
    model.train()
    clipper = AutoregressiveMasking()
    model.apply(clipper)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="sum")

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            #batch_data = data[indices]
            batch_data = data[indices].to(device)

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