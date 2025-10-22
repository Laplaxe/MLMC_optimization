import matplotlib.pyplot as plt

def set_plot_preferences():
    
    # Axes settings
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    
    
    # Tick settings
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # Legend settings
    plt.rcParams['legend.fontsize'] = 14
    
    # Font settings
   # plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Latin Modern']
    #plt.rcParams['text.usetex'] = True
    
    # Grid settings
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5