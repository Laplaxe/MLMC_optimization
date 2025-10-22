import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import math

class NadeLayer(nn.Module):
    """ nade layer """
    def __init__(self, size_in, N_h):
        super().__init__()
        self.size_in, self.N_h = size_in, N_h
        weights = torch.Tensor(N_h, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter. #????
        bias = torch.Tensor(N_h)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weights, a = -bound, b = bound) # weight init
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        data_len = x.shape[0]
        x = x.unsqueeze(2).repeat(1,1,self.size_in)
        #torch.cuda.empty_cache() #LMDB: Probably don't need to call it so often, but I was having troubles with RAM usage
        #start = time.time()
        x = x.triu(1) #autoregressive masking
        #torch.cuda.empty_cache()
        #end1 = time.time()
        x = torch.einsum("ij, kjl->kil",self.weights, x)
        #torch.cuda.empty_cache()
        #x = torch.matmul(self.weights, x)
        #end2 = time.time()
        #print((end2-end1)/(end1-start))
        x = torch.add(x, self.bias.reshape(1, self.N_h, 1))  # w times x + b
        torch.cuda.empty_cache()
        return x
class NadeFinalLayer(nn.Module):
    """ nade layer """
    def __init__(self, N, N_h):
        super().__init__()
        self.N, self.N_h = N, N_h
        weights = torch.Tensor(N_h, N)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter. #????
        bias = torch.Tensor(N)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weights, a = -bound, b = bound) # weight init
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        x = torch.mul(x, self.weights)
        #torch.cuda.empty_cache()
        x = torch.sum(x, axis = 1)
        torch.cuda.empty_cache()
        return torch.add(x, self.bias)  # w times x + b

# Build the autoregressive model
class nade(nn.Module):
    """Autoregressive Made"""
    def __init__(self, input_size, N_h):
        super(nade, self).__init__()
        self.layer = NadeLayer(input_size, N_h)
        self.activation = nn.Sigmoid()
        self.finallayer = NadeFinalLayer(input_size,N_h)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.finallayer(x)
        x = self.activation(x)
        return x

# Train the model
def train_nade(data, input_size, N_h = 64, epochs=50, batch_size=256, learning_rate=1e-3):
    """Train the made architecture using data"""
    model=nade(input_size, N_h = N_h)
    model = model.to("cuda")
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

    return model
