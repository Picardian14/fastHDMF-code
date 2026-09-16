#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the updated dyn_fic_DMF model with improved parameter handling.

This example shows:
1. How to configure and run the model with different plasticity modes
2. How to selectively return different outputs (E-rates, I-rates, BOLD, FIC)
3. How to use both scalar and vector parameters
4. How to visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt
from fastdyn_fic_dmf import DynFicDMF, PlasticityMode

def run_and_plot_example():
    """Run simulations with different parameters and plot the results."""
    # Create a simple 3-node network
    N = 3  # Number of brain regions
    
    # Create a simple connectivity matrix
    C = np.array([
        [0.0, 0.5, 0.2],
        [0.5, 0.0, 0.5],
        [0.2, 0.5, 0.0]
    ])
    
    # Create model instance
    model = DynFicDMF()
    
    # Get default parameters and modify for our example
    params = model.default_params(
        # Basic model parameters
        dt=0.1,              # Milliseconds
        dtt=1.0,             # Step size for BOLD integration
        TR=2000,             # Repetition time for BOLD (ms)
        G=2.0,               # Global coupling strength
        
        # Plasticity parameters - use vector format to demonstrate flexibility
        lr=np.array([0.1, 0.15, 0.12]),  # Node-specific learning rates
        taoj_time=np.array([10.0, 12.0, 8.0]),  # Node-specific decay times
        obj_rate=3.0,        # Target firing rate
        
        # Return flags - we want all outputs for this example
        return_rate_e=True,
        return_rate_i=True,
        return_bold=True,
        return_fic=True,
        
        # Set plasticity mode using the enum for clarity
        plasticity_mode=PlasticityMode.PLASTICITY_WITH_DECAY,
        
        # Optional seed for reproducibility
        seed=42
    )
    
    # Run simulation for 5 seconds (5000 ms)
    simulation_length = 5000  # ms
    steps = int(simulation_length / params['dt'])
    
    print("Running simulation with plasticity and decay...")
    results1 = model.run(params, C, steps)
    
    # Now run again with different plasticity mode
    params['plasticity_mode'] = PlasticityMode.PLASTICITY_WITHOUT_DECAY
    print("Running simulation with plasticity but no decay...")
    results2 = model.run(params, C, steps)
    
    # And once more with no plasticity
    params['plasticity_mode'] = PlasticityMode.NO_PLASTICITY
    print("Running simulation without plasticity...")
    results3 = model.run(params, C, steps)
    
    # Plot results
    plot_comparison(results1, results2, results3, params, steps)
    
    # Demonstrate returning only specific outputs
    print("\nDemonstrating selective output options...")
    
    # Only return excitatory rates (not inhibitory)
    params_e_only = params.copy()
    params_e_only['return_rate_e'] = True
    params_e_only['return_rate_i'] = False
    params_e_only['return_bold'] = False
    params_e_only['return_fic'] = False
    
    results_e_only = model.run(params_e_only, C, steps)
    print("Returned keys when only requesting E-rates:", list(results_e_only.keys()))
    
    # Only return BOLD and FIC
    params_bold_fic = params.copy()
    params_bold_fic['return_rate_e'] = False
    params_bold_fic['return_rate_i'] = False
    params_bold_fic['return_bold'] = True
    params_bold_fic['return_fic'] = True
    
    results_bold_fic = model.run(params_bold_fic, C, steps)
    print("Returned keys when only requesting BOLD and FIC:", list(results_bold_fic.keys()))

def plot_comparison(results1, results2, results3, params, steps):
    """Plot comparison of results from different plasticity modes."""
    time_ms = np.arange(steps) * params['dt']
    time_s = np.arange(results1['bold'].shape[1]) * params['TR'] / 1000
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Comparison of Different Plasticity Modes', fontsize=16)
    
    # Plot excitatory firing rates
    ax1 = fig.add_subplot(221)
    for i in range(3):
        ax1.plot(time_ms, results1['rate_e'][i], label=f'Node {i+1} (with decay)', linewidth=1.5)
        ax1.plot(time_ms, results2['rate_e'][i], '--', label=f'Node {i+1} (no decay)', linewidth=1.5)
        ax1.plot(time_ms, results3['rate_e'][i], ':', label=f'Node {i+1} (no plasticity)', linewidth=1.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Firing rate (Hz)')
    ax1.set_title('Excitatory Firing Rates')
    ax1.legend(loc='upper right')
    
    # Plot inhibitory firing rates
    ax2 = fig.add_subplot(222)
    for i in range(3):
        ax2.plot(time_ms, results1['rate_i'][i], label=f'Node {i+1} (with decay)')
        ax2.plot(time_ms, results2['rate_i'][i], '--', label=f'Node {i+1} (no decay)')
        ax2.plot(time_ms, results3['rate_i'][i], ':', label=f'Node {i+1} (no plasticity)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Firing rate (Hz)')
    ax2.set_title('Inhibitory Firing Rates')
    
    # Plot BOLD signals
    ax3 = fig.add_subplot(223)
    for i in range(3):
        ax3.plot(time_s, results1['bold'][i], label=f'Node {i+1} (with decay)')
        ax3.plot(time_s, results2['bold'][i], '--', label=f'Node {i+1} (no decay)')
        ax3.plot(time_s, results3['bold'][i], ':', label=f'Node {i+1} (no plasticity)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('BOLD signal')
    ax3.set_title('BOLD Signals')
    
    # Plot FIC values
    ax4 = fig.add_subplot(224)
    for i in range(3):
        ax4.plot(time_ms, results1['fic'][i], label=f'Node {i+1} (with decay)')
        ax4.plot(time_ms, results2['fic'][i], '--', label=f'Node {i+1} (no decay)')
        ax4.plot(time_ms, results3['fic'][i], ':', label=f'Node {i+1} (no plasticity)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('FIC value')
    ax4.set_title('Feedback Inhibition Control Values')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('dmf_plasticity_comparison.png')
    print("Figure saved as 'dmf_plasticity_comparison.png'")

def demonstrate_backward_compatibility():
    """Show how legacy code still works with the refactored API."""
    print("\nDemonstrating backward compatibility...")
    
    # Create model
    model = DynFicDMF()
    
    # Create a simple connectivity matrix
    N = 3
    C = np.eye(N) * 0.5
    
    # Old-style parameters
    params_legacy = {
        'C': C,
        'receptors': np.ones(N),
        'dt': 0.1,
        'batch_size': 1000,
        'TR': 2000,
        'dtt': 1.0,
        
        # Legacy parameters instead of new ones
        'lr_vector': np.ones(N) * 0.1,
        'taoj_vector': np.ones(N) * 10.0,
        'with_plasticity': True,
        'with_decay': True,
        'return_rate': True,
        'return_bold': True,
        'return_fic': True
    }
    
    # Run simulation
    steps = 1000
    results = model.run(params_legacy, nb_steps=steps)
    
    print("Legacy parameters converted successfully.")
    print("Available results:", list(results.keys()))

if __name__ == "__main__":
    run_and_plot_example()
    demonstrate_backward_compatibility() 