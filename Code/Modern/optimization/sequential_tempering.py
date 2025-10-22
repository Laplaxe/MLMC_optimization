import torch
import numpy as np
import sys
import argparse
import time

sys.path.append("../../../Code/Legacy/packages")
sys.path.append("../../../Code/Modern/optimization")

from geometry import *
from utilities import *
from data_loads import *
from monte_carlo import *
from made import *
from global_steps import *

from copy import deepcopy
import time

def MLMC_fast(model, data, beta, N, J,  num_steps = 10, return_correlations = False):
    # NOTA: le correlazioni sono calcolate usando come riferimento il primo campione
    # Non sono mediate sui tempi!
    #Se il primo campione è all'equilibrio, non ci sono problemi
    acc_rates = []
    if return_correlations:
        correlations = [1]
    with torch.no_grad():
        bce = nn.BCELoss(reduction = "none")
        current_config = data.clone()
        for t in range(num_steps):
            new_config = generate_config_fast(model, N, len(data), J)
            
            current_energy = compute_energy(current_config, J)
            current_probability = torch.sum(bce(model(current_config), (current_config+1)/2), axis = 1)

            new_energy = compute_energy(new_config, J)
            new_probability = torch.sum(bce(model(new_config), (new_config+1)/2), axis = 1)

            arg_new = -beta*new_energy + new_probability
            arg_current = -beta*current_energy + current_probability

            acceptances = (torch.log(torch.rand(size=(len(data),), device = "cuda")) < (arg_new-arg_current)).int()
            current_config = torch.einsum("i, ij->ij",(1-acceptances),current_config) + torch.einsum("i, ij->ij",acceptances, new_config)
            torch.cuda.empty_cache()
            acc_rates.append(torch.sum(acceptances)/len(data))
            if return_correlations:
                correlations.append(float(torch.mean(data*current_config) - torch.mean(data)*torch.mean(current_config)))
    if return_correlations:
        return current_config, acc_rates, correlations
    else:
        return current_config, acc_rates

def sequential_tempering(L, J, pop_size, num_steps_MC, swap_step, N, Tstart, Tend, Observables,
                                schedule = "Cv_beta", num_temps_determiner = 0.5, 
                                high_temp_thermalization_steps = 200, batch_size = 256, 
                                num_epochs_start = 40, num_epochs_retrain = 1, dimension = "3d"):

    #get the indices (needed for the checkerboard update)
    if dimension == "3d":
        even_indices, odd_indices = get_indices(L)
        N = L*L*L
    elif dimension == "2d":
        even_indices, odd_indices = get_indices_2D(L)
        N = L*L
    else:
        raise ValueError("dimension must be either 3d or 2d")

    #set the temperature schedule
    temperatures = schedule_temperatures(Tstart, Tend, num_temps_determiner, schedule, N)

    #initialize the population
    population = torch.randint(0, 2, (pop_size,N), device="cuda").float() * 2 - 1
    
    #initialize the observables
    observ = Observables(J, N)

    # Thermalize the high temperature population
    oldT = temperatures[0]
    for i in range(high_temp_thermalization_steps):
        population = monte_carlo_update_fast(population, J, beta=1/oldT, even_indices=even_indices, odd_indices=odd_indices)
    observ.update(population) #save the minimum and mean energie
    torch.cuda.synchronize()
    start_time = time.time()
    model = train_made_improved(population, N, epochs = num_epochs_start)
    for currT in temperatures[1:-1]:
        #Monte Carlo updates
        for i in range(num_steps_MC):
            population, _ = MLMC_fast(model, population, 1/currT, N, J, num_steps=1)
            for j in range(swap_step):
                population = monte_carlo_update_fast(population, J, 1/currT, even_indices, odd_indices)
        #retrain of the model
        model = retrain_made(model, population, epochs = num_epochs_retrain, batch_size=batch_size)
        observ.update(population)

    #for the last temperature, we do not need to retrain the model
    currT = temperatures[-1]
    for i in range(num_steps_MC):
        population, _ = MLMC_fast(model, population, 1/currT, N, J, num_steps=1)
        for j in range(swap_step):
            population = monte_carlo_update_fast(population, J, 1/currT, even_indices, odd_indices)
    observ.update(population)
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    return temperatures, observ, elapsed_time




#Running the parallel simulated annealing and parsing the arguments
if __name__ == "__main__":
    #parse all the possible arguments
    parser = argparse.ArgumentParser(description='Sequential Tempering')
    #general parsers
    parser.add_argument('--pop_size', type=int, default=10000, help='Population size')
    parser.add_argument('--L', type=int, default=10, help='Lattice size')
    parser.add_argument('--seed', type=int, default=310411727, help='Random seed')
    parser.add_argument('--Tstart', type=float, default=1.92, help='Starting temperature')
    parser.add_argument('--Tend', type=float, default=0.1, help='Ending temperature')
    parser.add_argument('--Cv_factor', type=float, default=1.618, help='Cv_factor')
    parser.add_argument('--MLMCsteps', type=int, default=5, help='Number of Machine-Learning assisted steps')
    parser.add_argument('--MCsteps', type=int, default=15, help='Number of Monte Carlo steps for each MLMC step')
    parser.add_argument('--num_temps', type=int, default=30, help='Number of annealing temperatures')
    parser.add_argument('--schedule', type=str, default="Cv_beta", help='Scheduling of temperatures')
    #specific parsers for training
    parser.add_argument('--num_epochs_start', type=int, default=40, help='Number of epochs for the first training')
    parser.add_argument('--num_epochs_retrain', type=int, default=1, help='Number of epochs for each temperature retraining')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    
    
    args = parser.parse_args()

    #check consistency for the schedules: Cv_beta requires Cv_factor, the others the number of steps
    if args.schedule != "Cv_beta" and args.Cv_factor != parser.get_default("Cv_factor"):
        parser.error("Cv_factor can only be specified when schedule is Cv_beta.")
    if args.schedule == "Cv_beta" and args.num_temps != parser.get_default("num_temps"):
        parser.error("num_temps cannot be specified when schedule is Cv_beta.")

    if args.schedule == "Cv_beta":
        num_temps_determiner = args.Cv_factor
    else:    
        num_temps_determiner = args.num_temps

    #define the parameters of the model
    pop_size = args.pop_size
    Tend = args.Tend
    Cv_factor = args.Cv_factor
    schedule = args.schedule
    L = args.L
    N = L*L*L
    seed = args.seed
    Tstart = args.Tstart
    J = read_couplings(f'../../../Data/Alpha/Couplings/couplings_L{L}_R1_seed{seed}.txt', N).cuda()
    Tstart = float(Tstart)
    Tend = float(Tend)

    MLMCsteps = args.MLMCsteps
    MCsteps = args.MCsteps

    num_epochs_start = args.num_epochs_start
    num_epochs_retrain = args.num_epochs_retrain
    batch_size = args.batch_size


    temperatures, observ, elapsed_time = sequential_tempering(L, J, pop_size, MLMCsteps, MCsteps,N,  Tstart,
                                                                                       Tend,  Observables, schedule = schedule,                                                                                                                                                             
                                                                                       num_epochs_start = num_epochs_start,
                                                                                       num_epochs_retrain = num_epochs_retrain,
                                                                                       batch_size = batch_size,
                                                                                       num_temps_determiner = num_temps_determiner)
    minimum = torch.tensor(observ.get_observable_history("min_energy")).min()
    print(MLMCsteps, MCsteps, f"{len(temperatures)}", schedule, f"{minimum:.5f}", f"{observ.get_observable_history('mean_energy')[-1]:.5f}", elapsed_time)