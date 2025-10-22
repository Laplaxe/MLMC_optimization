
import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch import optim as optim
import torch.nn as nn
from torch.nn import Parameter
from torch.optim.lr_scheduler import ExponentialLR

class N4_layer(MessagePassing):
    """Edge matrix, scalar sum of the fields"""
    def __init__(self, num_edges):
        super().__init__(aggr='add', node_dim=-2)  # "Add" aggregation (Step 5).
        self.weights = Parameter(torch.zeros(num_edges))
        torch.nn.init.normal_(self.weights, std=1.0/np.sqrt(num_edges))

    def forward(self, x, edge_index, weight_tensor):
        out = self.propagate(edge_index, x=x, weight_tensor=weight_tensor)
        return out

    def message(self, x_j, weight_tensor):
        return weight_tensor.view(-1, 1) * self.weights.view(-1, 1)*x_j
    
class N4(torch.nn.Module):
    def __init__(self, num_edges, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for k in range(num_layers):
            self.layers.append(N4_layer(num_edges))
                               

    def forward(self, h_0, edge_index, weight_tensor):
        #h_0 = x.float()
        #h = torch.clone(h_0)
        h = h_0
        for layer in self.layers[:-1]:
            h = layer(h, edge_index, weight_tensor)
            h = h + h_0
        h = self.layers[-1](h, edge_index, weight_tensor)
        h = torch.sigmoid(2*h)
        return h
    
def generate_config_gnn(model, N_spins, N_config, edge_index, weight_tensor, J, device = "cuda"):
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
        config = torch.zeros((N_config, N_spins)).to(device)
    
        # Generate each spin in an autoregressive manner
        for n in range(N_spins):

            #make the xi
            batch_data = torch.einsum("ik,kl-> il",config,J)
            batch_data = batch_data.T

            # Get probabilities from the autoregressive model for the nth spin
            probs = torch.squeeze(model(batch_data, edge_index, weight_tensor))
            
            # Sample new spin values based on probabilities and update the configuration
            config[:, n] = (torch.bernoulli(probs[n, :]) * 2 - 1)
            torch.cuda.empty_cache()
    
    return config

def train_N4(model, 
          datareal, 
          use_edges, 
          use_weights, 
          N, 
          batch_size = 100, 
          num_epochs = 60, 
          patience = 10, 
          learning_rate = 0.01, 
          scheduler_time = 10, 
          MIN_EPOCH_SCHEDULER = 0,
          PRINT_EVERY = 2,
          device = "cuda"):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #we use Adam optimizer
    criterion = nn.BCELoss(reduction="mean") #binary loss
    scheduler = ExponentialLR(optimizer, gamma=0.5) #scheduler

    best_loss = 100000000 #just to initialize
    for epoch in range(num_epochs):
        tot_loss = 0
        count = 0
        #get data and shuffle them
        datatrain = torch.clone(datareal)
        datatrain = datatrain[torch.randperm(datatrain.size(0))]
        
        for i in range(0, len(datatrain)-batch_size-1, batch_size): #iterate over batches
            
            #get the batch
            indices = torch.arange(i, i+batch_size).to(device)
            batcher = datatrain[indices]

            #some manipulations of the batch required for 
            batch_data = torch.einsum("ijk,kl-> ijl",batcher.repeat((N,1,1)).permute((1,0,2)).tril(diagonal = -1),J)
            batch_data = batch_data.view(-1,N)
            batch_data = batch_data.T

            #compute the field h
            h = model(batch_data, use_edges, use_weights)  # Perform a single forward pass.
            h = (h.T.view(batch_size,N,N)).diagonal(dim1=1, dim2=2)
            loss = criterion(h, (batcher+1.)/2.)  # Compute the loss solely based on the training nodes.

            
            tot_loss += loss #update the total loss
            count += 1 #count that we performed a step
            optimizer.zero_grad()  # Clear gradients.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        if epoch % PRINT_EVERY == 0: #print the loss every 2 steps
            print(f"Epoch {epoch} Loss = {float(tot_loss/count)}")

        if epoch % scheduler_time == 0 and epoch > MIN_EPOCH_SCHEDULER: #perform the schedulling step
            scheduler.step()

        #save the model if its the best loss. In principle should be done on the evaluation set, 
        # but if we assume that the data found using parallel tempering are good enough (e.g. they are rapresentative enough)
        # it should not be needed
        if tot_loss/count < best_loss:
            best_loss = tot_loss/count
            epochs_since_best_val_acc = 0
            best_weights = model.state_dict()
        else:
            epochs_since_best_val_acc += 1

        # load the best model after patience
        if epochs_since_best_val_acc >= patience:
            model.load_state_dict(best_weights)
            break
    # load the best model after the fixed number of epochs
    model.load_state_dict(best_weights)