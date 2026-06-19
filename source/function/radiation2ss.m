function radiation_state_space = radiation2ss(nbr_state, omega, added_mass, damping, added_mass_inf, graph)
% Name:            Radiation state space modeling
% Description:     Model the radiation force as a linear state space with a
%                  given number of unmeasurable state.
%
% Inputs:
%    input1 - Description of the first input (e.g., data type, dimensions).
%    input2 - Description of the second input.
%
% Outputs:
%    output1 - Description of the first output.
%    output2 - Description of the second output.
%
% Author:        Antoine Dupuis
% Collaborator:  
% Date:          January 2025
% Project:       - MPC course project
%                - EWTEC 2025: wave to wire MPC
%
% Dependencies:
% - [Dependency 1]: Description of any functions, toolboxes, or files required.
% -------------------------------------------------------------------------

fit_ini = 1000000000;

% dealing with optional input
if nargin < 6
        graph = 'no';
end

% # of states
if length(nbr_state) == 1
    nbr_state = 1:nbr_state;
end

n = length(nbr_state);                                                     % number of case

% compute the discrete tf
H_ref = sqrt(-1) .* omega .* (added_mass - added_mass_inf) + damping;

% initialize fitting criterion for tf
fit = fit_ini;

% for every state #
for i = 1:n

    % order of the rational function
    order_D = nbr_state(i);                                               
    order_N = order_D - 1;

    % determine coefficient of the rational tf
    tol = [];
    [b,a] = invfreqs(H_ref, omega, order_N, order_D, tol, 30);             % /!\ reflect tolerance

    % built tf
    H_fit = tf(b, a);

    % check on stability (if all poles have negative real parts)
    poles = pole(H_fit);
    is_stable = all(real(poles) < 0);

    if is_stable == 1

        % compute discrete fitted tranfert function over omega             % /!\ what about phase ?
        h_fit = squeeze(freqresp(H_fit, omega));
        
        % compute fitting
        mse = metricError(abs(h_fit), abs(H_ref), 'mse');

        % store if better
        if mse < fit
            H_rad = H_fit;
            h_rad = h_fit;
            fit = mse;
            n_states = nbr_state(i);
        end
    end
end

% derive state-space from tf
ss_rad = ss(H_fit);

% verify that a solution was found: stop if not, store if yes
if fit == fit_ini
        error('no stable model was found.');
else
    % function output
    radiation_state_space = ss_rad;
end

% plot if requested
if strcmp(graph, 'plot')
    
    % show state space
    [A, B, C, D] = tf2ss(b, a);

    % plot h_rad against H_ref
    figure;
    plot(omega, abs(H_ref))
    hold on 
    plot(omega, abs(h_rad), '--')
    xlabel('\omega')
    ylabel('|H|')
    grid on
    legend('original', 'fitted')
    title(['Comparison between original and best fitted trasnfert function: state #: ', num2str(n_states), ', MSE = ', num2str(fit)])
end
    
end
