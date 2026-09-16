function [ params ] = dyn_fic_DefaultParams(varargin)
%%DEFAULTPARAMS Default parameter settings for DMF simulation
%
%   P = DEFAULTPARAMS() yields a struct with default values for all necessary
%   parameters for the DMF model. The default structural connectivity is the
%   DTI fiber consensus matrix obtained from the HCP dataset, using a
%   Schaeffer-100 parcellation.
%
%   P = DEFAULTPARAMS('key', 'value', ...) adds (or replaces) field 'key' in
%   P with given value.
%
% Pedro Mediano, Feb 2021

params = [];

% Connectivity matrix
if any(strcmp(varargin, 'C'))
  C = [];

else
  try
    p = strrep(mfilename('fullpath'), 'dyn_fic_DefaultParams', '');
    C = dlmread([p, '../data/DTI_fiber_consensus_HCP.csv'], ',');
    C = C/max(C(:));  
  catch
    error('No connectivity matrix provided, and default matrix not found.');
  end

end


% DMF parameters
params.C         = C;
params.receptors = 0;
params.dt        = 0.1;     % ms
params.taon      = 100;     % NMDA tau ms
params.taog      = 10;      % GABA tau ms
params.gamma     = 0.641;   % Kinetic Parameter of Excitation
params.sigma     = 0.01;    % Noise SD nA
params.JN        = 0.15;    % excitatory synaptic coupling nA
params.I0        = 0.382;   % effective external input nA
params.Jexte     = 1.;      % external->E coupling
params.Jexti     = 0.7;     % external->I coupling
params.w         = 1.4;     % local excitatory recurrence
params.g_e       = 0.16;    % excitatory conductance
params.Ie        = 125/310; % excitatory threshold for nonlineariy
params.ce        = 310.;    % excitatory non linear shape parameter
params.g_i       = 0.087;   % inhibitory conductance
params.Ii        = 177/615; % inhibitory threshold for nonlineariy
params.ci        = 615.;    % inhibitory non linear shape parameter
params.wgaine    = 0;       % neuromodulatory gain
params.wgaini    = 0;       % neuromodulatory gain
params.G         = 2;       % Global Coupling Parameter

% Dynamic Fic Parameters
params.LR       = 1;       % FIC Learning rate
params.DECAY      = 50000;       % FIC decay constant
params.obj_rate  = 3.44;       % FIC objective firing rate
params.return_rate=true;
params.return_bold=true;
params.return_fic=false;
params.with_plasticity=true;
params.with_decay=true;

% Balloon-Windkessel parameters (from firing rates to BOLD signal)
params.TR  = 2;     % number of seconds to sample bold signal
params.dtt = 0.001; % BW integration step, in seconds
% Neurovascular coupling parameters
params.nvc_sigmoid = true; % use sigmoid to convert firing rates to vasodilatory signal
params.nvc_match_slope = false; % if using sigmoid, match slope at obj_rate
params.nvc_r0         = params.obj_rate;    % baseline firing-rate (Hz). If your r is already baseline-subtracted, set 0.0
params.nvc_u50         = 15.0;   % half-saturation (Hz): compression starts in 15–30 Hz range
params.nvc_k          = 0.25;    % maximum vasodilatory signal gain

% Parallel computation parameters
params.batch_size = 5000;

% Add/replace remaining parameters
for i=1:2:length(varargin)
  params.(varargin{i}) = varargin{i+1};
end

% If feedback inhibitory control not provided, use heuristic
if ~any(strcmp(varargin, 'J'))
  params.J = 0.75*params.G.*sum(params.C, 1)' + 1;
end

% if decay or lr is not provided make as a vector yourself
if params.with_decay && length(params.taoj) == 1
    params.taoj = params.taoj*ones(size(params.C,1),1); 
end
if params.with_plasticity && length(params.lrj) == 1
    params.lrj = params.lrj*ones(size(params.C,1),1);
end

end

