%%DMF Run simulation of Dynamic Mean Field model of brain dynamics.
%
%   B = dyn_fic_DMF(params, T);
%     Run model for T timesteps and returns simulated BOLD time series.
%
%   R = dyn_fic_DMF(params, T, 'rate');
%   B = dyn_fic_DMF(params, T, 'bold');
%   [R, B] = DMF(params, nb_steps, 'both');
%     Run model for T timesteps and returns time series of firing rates, BOLD
%     activity, or both.
%
% This is a C++ MEX-based implementation that needs to be compiled to be used.
% See README file for details.
%
% NOTE: Computing (and returning) both rates and BOLD has a time and memory
% overhead.  For optimal performance, avoid the 'both' option if possible.
%
% Pedro Mediano, Apr 2020
%

