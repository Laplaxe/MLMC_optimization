Code and data for the paper "Demonstrating real advantage of machine learning--enhanced Monte Carlo for combinatorial optimization".

- The Code directory contains code to run the optimization algorithms. In particular, the Legacy subdirectory contains supporting code, while Modern/optimization contains the code to perform Simulated, Population and Global Annealing.
- The Data directory contains the data supporting the results in the paper. The Alpha subdirectory contains the spin interaction files (saved as "`<index_spin_1`> `<index_spin_2`> `<interaction`>") and the Gurobi logs, while the Omega subdirectory contains the results of the runs of the annealings: each row of the file contains a different run and is in the form
  - `<MCS_per_temperature`> `<number_of_temperatures`> `<schedule`> `<minimum_energy_found`> `<average_energy_at_T=0.1`> `<runtime_in_seconds`> (for Simulated Annealing and Population Annealing)
  - `<global_steps_per_temperature`> `<MCS_per_global_steps`> `<number_of_temperatures`> `<schedule`> `<minimum_energy_found`> `<average_energy_at_T=0.1`> `<runtime_in_seconds`> (for Global Annealing).
- The Plots directory contains the both the code for producing the plots and the plots themselves.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex

@article{del2026demonstrating,
  title={Demonstrating real advantage of machine learning--enhanced Monte Carlo for combinatorial optimization},
  author={Del Bono, Luca Maria and Ricci-Tersenghi, Federico and Zamponi, Francesco},
  journal={Proceedings of the National Academy of Sciences},
  volume={123},
  number={19},
  pages={e2534768123},
  year={2026},
  publisher={National Academy of Sciences}
}

