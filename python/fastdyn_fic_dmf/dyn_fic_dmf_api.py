"""
Dynamic Mean Field model

Run simulation of the Dynamic Mean Field model of brain dynamics.

Pedro Mediano, Jun 2020
"""
import _DYN_FIC_DMF
import numpy as np

__all__ = ['default_params', 'run']

def _format_dict(d):
    """
    Makes sure that every value in the dictionary is a np.array
    
    Parameters
    ----------
    d : dict
        Parameter dictionary with strings as keys.

    Returns
    -------
    q : dict
        Parameter dictionary with strings as keys and np.ndarrays as values.
    """
    q = {}
    for k in d:
        if isinstance(d[k], np.ndarray):
            if not d[k].shape:
                q[k] = d[k].reshape(1).astype(float)
            else:
                q[k] = d[k].astype(float)
        elif isinstance(d[k], (tuple, list)):
            q[k] = np.array(d[k], dtype=float)
        elif np.isscalar(d[k]):
            q[k] = np.array([d[k]], dtype=float)
        else:
            raise ValueError("Parameter %s cannot be cast as float np.array"%s)
    return q


def default_params(**kwargs):
    """
    Default parameters for DMF simulation.

    Parameters
    ----------
    kwargs
        Name-value pairs to add to or replace in the dictionary.

    Returns
    -------
    params: dict
    """

    if 'C' not in kwargs:
        C = np.loadtxt(__file__.rstrip('dyn_fic_dmf_api.py') + 'DTI_fiber_consensus_HCP.csv', delimiter=',')
        C = C/C.max()
    else:
        C = []


    # DMF parameters
    params              = {}
    params['C']         = C        # structural connectivity
    params['receptors'] = 0        # receptor density
    params['dt']        = 0.1      # ms
    params['taon']      = 100      # NMDA tau ms
    params['taog']      = 10       # GABA tau ms
    params['gamma']     = 0.641    # Kinetic Parameter of Excitation
    params['sigma']     = 0.01     # Noise SD nA
    params['JN']        = 0.15     # excitatory synaptic coupling nA
    params['I0']        = 0.382    # effective external input nA
    params['Jexte']     = 1.       # external->E coupling
    params['Jexti']     = 0.7      # external->I coupling
    params['w']         = 1.4      # local excitatory recurrence
    params['g_e']        = 0.16     # excitatory non linear shape parameter
    params['Ie']        = 125/310  # excitatory threshold for nonlinearity
    params['ce']       = 310.     # excitatory conductance
    params['g_i']        = 0.087    # inhibitory non linear shape parameter
    params['Ii']        = 177/615  # inhibitory threshold for nonlinearity
    params['ci']       = 615.     # inhibitory conductance
    params['wgaine']    = 0        # neuromodulatory gain
    params['wgaini']    = 0        # neuromodulatory gain
    params['lr_scaling'] = 0
    params['G']         = 2        # Global Coupling Parameter
    
    # Dynamic Fic options options
    params['lrj']       = 1
    params['taoj']       = 50000
    params['obj_rate']       = 3.44
    params["return_bold"] = True
    params["return_rate"] = False
    params["return_fic"] = False
    params["with_decay"] = True
    params["with_plasticity"] = True
    # Balloon-Windkessel parameters (from firing rates to BOLD signal)
    params['TR']  = 2     # number of seconds to sample bold signal
    params['dtt'] = 0.001 # BW integration step, in seconds
    # Neurovascular coupling parameters
    params['nvc_sigmoid'] = False # use sigmoid to convert firing rates to vascular signal
    params['nvc_match_slope'] = True # if using sigmoid, match slope at obj_rate
    params['nvc_r0']         =  params['obj_rate']  # baseline firing-rate
    params['nvc_u50']         = 15.0   # half-saturation (Hz): compression starts in 15–30 Hz range
    params['nvc_k']          = 0.25    # maximum vasodilatory signal gain

    # Parallel computation parameters
    params['batch_size'] = 5000
    params['burnout'] = 5
    # Add/replace remaining parameters
    for k, v in kwargs.items():
        params[k] = v

    # If feedback inhibitory control not provided, use heuristic
    if 'J' not in kwargs:
        params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1

    # if decay or lr is not provided make as a vector yourself
    if params['with_decay'] and np.isscalar(params['taoj']):
        params['taoj'] = params['taoj']*np.ones(params['C'].shape[0]) 
    if params['with_plasticity'] and np.isscalar(params['lrj']):
        params['lrj'] = params['lrj']*np.ones(params['C'].shape[0])
    return params


def run(params, nb_steps):
    """
    Run the DMF model and return simulated brain activity. Size and number of
    output arguments depends on desired_out.

    Parameters
    ----------
    params : dict
        Parameter dictionary (see default_params()).

    nb_steps : int
        Number of integration steps to compute. Final size of the simulated
        time series depends on selected dt and TR.
    Returns
    -------
    out : list
        Simulated activity of the DMF model. Returns excitatory rates, inhibitory rates, bold activity and average FIC time series
    """

    # Pre-allocate memory for results
    N = params['C'].shape[0]
    nb_steps_bold = round(nb_steps*params['dtt']/params['TR'])
    if params["return_rate"]:
        nb_steps_rate = nb_steps
    else:
        nb_steps_rate = 2*params['batch_size']
    if params["return_fic"]:
        nb_steps_fic = nb_steps
    else:
        nb_steps_fic = 2*params['batch_size']
    
    rate_e_res = np.zeros((N, nb_steps_rate), dtype=float, order='F')
    rate_i_res = np.zeros((N, nb_steps_rate), dtype=float, order='F')
    bold_res = np.zeros((N, nb_steps_bold), dtype=float, order='F')
    fic_res = np.zeros((N, nb_steps_fic), dtype=float, order='F')


    # Run simulation
    sim = _DYN_FIC_DMF.DYN_FIC_DMF(_format_dict(params), nb_steps, N)    
    sim.run(rate_e_res, rate_i_res,bold_res, fic_res)


    return rate_e_res, rate_i_res, bold_res, fic_res
