"""
Dynamic Mean Field model with FIC (Feedback Inhibition Control)

The fastdyn_fic_dmf package provides tools for simulating brain dynamics
using the Dynamic Mean Field model with feedback inhibition control.
"""

from .dyn_fic_dmf_api import default_params, run

__all__ = ['DynFicDMF', 'default_params', 'run']

class DynFicDMF:
    """
    Dynamic Mean Field Model with Feedback Inhibition Control.
    
    This class provides a simple interface for running DMF simulations
    with various options for plasticity and output formats.
    """
    
    def __init__(self):
        """Initialize the DMF model."""
        pass
    
    def run(self, params, C=None, nb_steps=10000):
        """
        Run the DMF simulation with the given parameters.
        
        Parameters
        ----------
        params : dict
            Parameter dictionary (see default_params()).
            
        C : ndarray, optional
            Connectivity matrix. If provided, overrides the one in params.
            
        nb_steps : int, optional
            Number of integration steps to compute. Default is 10000.
            
        Returns
        -------
        results : dict
            Dictionary with simulation results based on the return flags:
            - 'rate_e': Excitatory firing rates (if return_rate_e is True)
            - 'rate_i': Inhibitory firing rates (if return_rate_i is True)
            - 'bold': BOLD signals (if return_bold is True)
            - 'fic': Feedback inhibition control values (if return_fic is True)
        """
        # Make a copy of params to avoid modifying the original
        params_copy = params.copy()
        
        # Update connectivity matrix if provided
        if C is not None:
            params_copy['C'] = C
        
        # Run the simulation using the API function
        return run(params_copy, nb_steps)
    
    def default_params(self, **kwargs):
        """
        Get default parameters for the DMF model.
        
        Parameters
        ----------
        **kwargs
            Name-value pairs to override default parameters.
            
        Returns
        -------
        params : dict
            Dictionary of model parameters.
        """
        return default_params(**kwargs)
