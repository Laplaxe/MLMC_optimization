import sys
import numpy as np
import torch
import networkx as nx
import sys
import pandas as pd
import random
from geometry import *
from utilities import *

def get_data(L: int, T: float, seed: int, EVERY = 1, RUN = 1, device = "cuda", back = "..", ordering = "spiral") -> tuple: 
    """Load data (configurations and J)
    Inputs:
    - L (int): side dimension of the system
    - T (float): temperature of the system
    - seed (int): seed used.
    - EVERY (int): take every EVERY-th configuration
    - RUN (int): which run to take data from
    - device (str): device to use
    - back (str): path to the dati_PT folder)
    - ordering (str): ordering of the spins. Can be 'spiral' or 'raster'
    Returns tuple containing:
    - data: configurations data
    - J: coupling data
    - use_edges: edges indexing (start and end spin of edge)
    - use_weights: edge weights (as given by J)
    in the spiral_contour order."""
    
    #First load the data
    data = pd.read_csv(f"{back}/dati_PT/run{RUN}_L{L}/repeat/L"+str(L)+"_S"+str(seed)+"/temperature_"+str(T)+".txt", delimiter=' ', comment='#', header=None)
    data = np.array(data)[:,:-1] #pandas adds an additional column due to additional space at the end of every row
    data = data[::EVERY] #let's try to use a subset of data
    np.random.shuffle(data)
    J = np.loadtxt(f"{back}/dati_PT/hard_instances_couplings/L{L}/L"+str(L)+"_S"+str(seed)+".txt") #Load interaction
    data = torch.Tensor(data)
    J = torch.Tensor(J)

    #get the order
    if ordering == "spiral":
        order = get_patch(0, L, L)[1]
    elif ordering == "raster":
        order = np.arange(L*L)
    else:
        raise  ValueError(f"{ordering} ordering is not supported. Valid ordering are 'spiral' and 'raster'.")
    #lattice = nx.grid_2d_graph(L,L)
    #lattice = nx.adjacency_matrix(lattice).toarray()
    #lattice = lattice[order][:,order]
    #lattice = np.triu(lattice, k=1).T
    data = data[:, order]
    #lattice = torch.Tensor(lattice).cuda()

    Jmat = J[order][:,order]
    J = Jmat

    edges = torch.nonzero(J)
    #edgecopy = torch.clone(edges)
    #edges = edgecopy[:, [1, 0]]
    edges = edges.T

    edge_weight = J[edges.T[:,0], edges.T[:,1]]

    #switch_edge = torch.clone(edges)
    #switch_edge[0], switch_edge[1] = edges[1], edges[0]
    #use_edges = torch.cat((edges, switch_edge, torch.arange(L*L).repeat(2, 1).to(device)), dim = 1)
    use_edges = torch.cat((edges, torch.arange(L*L).repeat(2, 1)), dim = 1)
    
    #use_weights = torch.cat((edge_weight, edge_weight, torch.ones(L*L)))
    use_weights = torch.cat((edge_weight, torch.ones(L*L)))

    data = data.to(device)
    J = J.to(device)
    use_edges = use_edges.to(device)
    use_weights = use_weights.to(device)


    return data, J, use_edges, use_weights

def get_data_old(L: int, T: float, seed: int, EVERY = 1, device = "cuda", back = "..") -> tuple: 
    """Load data (configurations and J)
    Inputs:
    -L (int): side dimension of the system
    -T (float): temperature of the system
    -seed (int): seed used.
    Returns tuple containing:
    -data: configurations data
    -J: coupling data
    -use_edges: edges indexing (start and end spin of edge)
    -use_weights: edge weights (as given by J)
    in the spiral_contour order."""
    
    #First load the data
    data = pd.read_csv(f"{back}/dati_PT/data_L16_old/run1/repeat/L"+str(L)+"_S"+str(seed)+"/temperature_"+str(T)+".txt", delimiter=' ', comment='#', header=None)
    data = np.array(data)[:,:-1] #pandas adds an additional column due to additional space at the end of every row
    data = data[::EVERY] #let's try to use a subset of data
    np.random.shuffle(data)
    J = np.loadtxt(f"{back}/dati_PT/hard_instances_couplings/J_L"+str(L)+"_S"+str(seed)+".dat") #Load interaction
    data = torch.Tensor(data)
    J = torch.Tensor(J)

    #get the order
    order = get_patch(0, L, L)[1]
    #lattice = nx.grid_2d_graph(L,L)
    #lattice = nx.adjacency_matrix(lattice).toarray()
    #lattice = lattice[order][:,order]
    #lattice = np.triu(lattice, k=1).T
    data = data[:, order]
    #lattice = torch.Tensor(lattice).cuda()

    Jmat = J[order][:,order]
    J = Jmat

    edges = torch.nonzero(J)
    #edgecopy = torch.clone(edges)
    #edges = edgecopy[:, [1, 0]]
    edges = edges.T

    edge_weight = J[edges.T[:,0], edges.T[:,1]]

    #switch_edge = torch.clone(edges)
    #switch_edge[0], switch_edge[1] = edges[1], edges[0]
    #use_edges = torch.cat((edges, switch_edge, torch.arange(L*L).repeat(2, 1).to(device)), dim = 1)
    use_edges = torch.cat((edges, torch.arange(L*L).repeat(2, 1)), dim = 1)
    
    #use_weights = torch.cat((edge_weight, edge_weight, torch.ones(L*L)))
    use_weights = torch.cat((edge_weight, torch.ones(L*L)))

    data = data.to(device)
    J = J.to(device)
    use_edges = use_edges.to(device)
    use_weights = use_weights.to(device)


    return data, J, use_edges, use_weights

def get_data_from_run2_old(L: int, T: float, seed: int, EVERY = 1, device = "cuda", back = "..") -> tuple: 
    """Load data (configurations and J)
    Inputs:
    -L (int): side dimension of the system
    -T (float): temperature of the system
    -seed (int): seed used.
    Returns tuple containing:
    -data: configurations data
    -J: coupling data
    -use_edges: edges indexing (start and end spin of edge)
    -use_weights: edge weights (as given by J)
    in the spiral_contour order."""
    
    #First load the data
    data = pd.read_csv(f"{back}/dati_PT/data_L16_old/run1/repeat/L"+str(L)+"_S"+str(seed)+"/temperature_"+str(T)+".txt", delimiter=' ', comment='#', header=None)
    data = np.array(data)[:,:-1] #pandas adds an additional column due to additional space at the end of every row
    data = data[::EVERY] #let's try to use a subset of data
    np.random.shuffle(data)
    J = np.loadtxt(f"{back}/dati_PT/hard_instances_couplings/J_L"+str(L)+"_S"+str(seed)+".dat") #Load interaction
    data = torch.Tensor(data)
    J = torch.Tensor(J)

    #get the order
    order = get_patch(0, L, L)[1]
    #lattice = nx.grid_2d_graph(L,L)
    #lattice = nx.adjacency_matrix(lattice).toarray()
    #lattice = lattice[order][:,order]
    #lattice = np.triu(lattice, k=1).T
    data = data[:, order]
    #lattice = torch.Tensor(lattice).cuda()

    Jmat = J[order][:,order]
    J = Jmat

    edges = torch.nonzero(J)
    #edgecopy = torch.clone(edges)
    #edges = edgecopy[:, [1, 0]]
    edges = edges.T

    edge_weight = J[edges.T[:,0], edges.T[:,1]]

    #switch_edge = torch.clone(edges)
    #switch_edge[0], switch_edge[1] = edges[1], edges[0]
    #use_edges = torch.cat((edges, switch_edge, torch.arange(L*L).repeat(2, 1).to(device)), dim = 1)
    use_edges = torch.cat((edges, torch.arange(L*L).repeat(2, 1)), dim = 1)
    
    #use_weights = torch.cat((edge_weight, edge_weight, torch.ones(L*L)))
    use_weights = torch.cat((edge_weight, torch.ones(L*L)))

    data = data.to(device)
    J = J.to(device)
    use_edges = use_edges.to(device)
    use_weights = use_weights.to(device)


    return data, J, use_edges, use_weights

def get_data_from_run2(L: int, T: float, seed: int, EVERY = 1, device = "cuda", back = "..") -> tuple: 
    """Load data (configurations and J)
    Inputs:
    -L (int): side dimension of the system
    -T (float): temperature of the system
    -seed (int): seed used.
    Returns tuple containing:
    -data: configurations data
    -J: coupling data
    -use_edges: edges indexing (start and end spin of edge)
    -use_weights: edge weights (as given by J)
    in the spiral_contour order."""
    
    #First load the data
    data = pd.read_csv(f"{back}/dati_PT/run2_L{L}/repeat/L"+str(L)+"_S"+str(seed)+"/temperature_"+str(T)+".txt", delimiter=' ', comment='#', header=None)
    data = np.array(data)[:,:-1] #pandas adds an additional column due to additional space at the end of every row
    data = data[::EVERY] #let's try to use a subset of data
    np.random.shuffle(data)
    J = np.loadtxt(f"{back}/dati_PT/hard_instances_couplings/L{L}/L"+str(L)+"_S"+str(seed)+".txt") #Load interaction
    data = torch.Tensor(data)
    J = torch.Tensor(J)

    #get the order
    order = get_patch(0, L, L)[1]
    #lattice = nx.grid_2d_graph(L,L)
    #lattice = nx.adjacency_matrix(lattice).toarray()
    #lattice = lattice[order][:,order]
    #lattice = np.triu(lattice, k=1).T
    data = data[:, order]
    #lattice = torch.Tensor(lattice).cuda()

    Jmat = J[order][:,order]
    J = Jmat

    edges = torch.nonzero(J)
    #edgecopy = torch.clone(edges)
    #edges = edgecopy[:, [1, 0]]
    edges = edges.T

    edge_weight = J[edges.T[:,0], edges.T[:,1]]

    #switch_edge = torch.clone(edges)
    #switch_edge[0], switch_edge[1] = edges[1], edges[0]
    #use_edges = torch.cat((edges, switch_edge, torch.arange(L*L).repeat(2, 1).to(device)), dim = 1)
    use_edges = torch.cat((edges, torch.arange(L*L).repeat(2, 1)), dim = 1)
    
    #use_weights = torch.cat((edge_weight, edge_weight, torch.ones(L*L)))
    use_weights = torch.cat((edge_weight, torch.ones(L*L)))

    data = data.to(device)
    J = J.to(device)
    use_edges = use_edges.to(device)
    use_weights = use_weights.to(device)


    return data, J, use_edges, use_weights

def get_raw_data(L,T, seed, EVERY = 1, RUN = 1, back = ".."):
    data = pd.read_csv(f"{back}/dati_PT/run{RUN}_L{L}/repeat/L{L}_S{seed}/temperature_"+str(T)+".txt", delimiter=' ', comment='#', header=None)
    data = np.array(data)[:,:-1] #pandas adds an additional column due to additional space at the end of every row
    data = data[::EVERY] #let's try to use a subset of data
    np.random.shuffle(data)
    data = torch.Tensor(data)
    return data

def get_raw_data_old(L,T, seed, EVERY = 1, RUN = 1, back = ".."):
    data = pd.read_csv(f"{back}/dati_PT/data_L16_old/run{RUN}/repeat/L{L}_S{seed}/temperature_"+str(T)+".txt", delimiter=' ', comment='#', header=None)
    data = np.array(data)[:,:-1] #pandas adds an additional column due to additional space at the end of every row
    data = data[::EVERY] #let's try to use a subset of data
    np.random.shuffle(data)
    data = torch.Tensor(data)
    return data