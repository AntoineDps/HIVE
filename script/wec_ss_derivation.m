% -------------------------------------------------------------------------
% Name:            WEC state space derivation
% Description:     Derive the linear state space model of a WEC, with
%                  radiation force as state space sub-system.
%                  Compare with experimental results.
%
% Author:          Antoine Dupuis
% Collaborator:    Vinicius 
% Date created:    January 2025
% Data:            - linear PTO plymouth run (for validation)
%                  - xModel simulation: hyd coefficients from WAMIT
%                  - xModel simulation: simulation reference
% Project:         - MPC course project
%                  - EWTEC 2025: wave to wire MPC
%
% Inputs: 
% - buoy.mat    : hyd. coef data
% - response.mat: ode4 based simulation
% - WTrun.mat   : experimental data for validation.
%
% Outputs:
% - model_name.mat: linear state space full wec (included radiation) object
%
% Dependencies:
% - radiation2ss.m: function to compute the radiation state space subsystem
% -------------------------------------------------------------------------

close all
clear all 
clc 

%% TO DO

% problem sign radiation
% reduce matrix size input (remove 0)

%% SET UP SCRIPT

% make local paths
myfodler = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(myfodler, 'source')));

% make path output
outputFolder = fullfile(myfodler, 'models');

%% INPUT

% hyd. data name 
hydfile = 'EllipsoidPlym_scaled_1b_B49_out';
% hydfile = 'UU_fullscale_B57_0p02_out';

% experimental object
expfile = 'B49.mat';

% radiation approximation
n_state = [3];

% wec model name
model_name = 'ss_ellipsoid';

%% LOAD & SORT

% load hyd data
load(fullfile('data', hydfile, 'buoy.mat'))
load(fullfile('data', hydfile, 'response.mat'))

% load experimental data
load(fullfile('data', expfile))

% get wec parameters
m     = buoy.mass + buoy.pto.mass;
r     = buoy.radius;
draft = buoy.draft;
k     = buoy.stiffness;                                                    % hydrodynamic stiffness
gamma = buoy.pto.gamma;                                                    % pto damping  

% get hyd. coef
ma_inf = buoy.addedMass33Inf;                                            
ma     = buoy.addedMass33;
B      = buoy.radiationRes33;
omega  = buoy.angularFreq;

% load ode4 values
x_ode4   = response.z(:,1); 
v_ode4   = response.z(:,2); 
eta_ode4 = response.eta;
Fe       = response.F(:,1);
Fr_ode   = response.F(:,2);
t        = response.t; 
dt       = t(2) - t(1);

% get experimental values
x_exp    = WTrun.buoy33(1:length(t));  
eta_exp  = WTrun.eta1_filt(1:length(t));  

%% RADIATION SUBSYSTEM

% compute radiation state space
rad_ss = radiation2ss(n_state, omega, ma, B, ma_inf, 'plot');

% get radiation state space matrices
A_r = rad_ss.A;
       
B_r = rad_ss.B;

C_r = rad_ss.C;

D_r = rad_ss.D;

% # of radiation state
n = length(A_r(:,1));

% check radiation force
[Fr, t, x] = lsim(rad_ss, v_ode4, t);                                      % /!\ problem sign !

figure;
plot(t, Fr')
hold on
plot(t, Fr_ode, '--')
xlabel('time')
ylabel('F_r')
title('Radiation force state space check')
legend('linear state space', 'ode4')

%% WEC LINEAR STATE SPACE

% define the state-space matrices for the WEC (without radiation force)
A_wec = [0             1;
         -k/(m+ma_inf) 0;];

B_wec = [0            0;
         1/(m+ma_inf) 1/(m+ma_inf)];

C_wec = [1 0
         0 1];

D_wec = [0 0
         0 0];

wec_SS = ss(A_wec, B_wec, C_wec, D_wec);

% merge radiation and full wec 
A_full = [A_wec(1,:) , zeros(1, n)     ;
          A_wec(2,:) , -C_r./(m+ma_inf);                                   % -C_r because the radiation force is withdrawn
          zeros(n, 1), B_r, A_r       ];

B_full = [B_wec(1,:) , zeros(1, n);
          B_wec(2,:) , zeros(1, n);
          zeros(n, n+2)           ];

C_full = [C_wec(1,:), zeros(1, n);
          C_wec(2,:), zeros(1, n);
          0,       0, C_r        ];

D_full = zeros(3, n+2);

full_wec_ss = ss(A_full, B_full, C_full, D_full);

%% VALIDATION

% initial states
x_0 = 0;
v_0 = 0;

X0 = [x_0 v_0 zeros(1,n)]';

% pre-allocate arrays for storing results
X  = zeros(length(t), n+2);   
Fr = zeros(length(t), 1);

% Initialize state vector
X_i = X0; 
 
% run state space model
for i = 1:length(t)-1

    % compute f_pto and radiation force
    F_pto = -gamma * X_i(2); 
    Fr(i) = C_full(end, :) * X_i;

    % store states and radiation force
    X(i, :) = X_i';

    % Compute the next state using the state-space equations
    % State derivative: dx = A * X + B * [Fe; Fpto]
    dX = A_full * X_i + B_full * [Fe(i); F_pto; zeros(n, 1)]; 

    % Update the state - Euler method
    X_i = X_i + dX * dt;
    
end

% position as outout
x_pred = X(:,1);

% plot response
figure;
plot(t, eta_ode4)
hold on
plot(t, eta_exp, '--')
hold on
plot(t, x_exp, 'black')
hold on
plot(t, x_pred, 'blue--')
hold on
plot(t, x_ode4, 'red--')
xlabel('time')
ylabel('position')
title('Linear state space WEC model validation')
legend('eta simulation', 'eta experimental', 'experimental', 'linear state space', 'ode4')

% plot radiation force
figure;
plot(t, Fr')
hold on
plot(t, Fr_ode, '--')
xlabel('time')
ylabel('F_r')
title('Radiation force check in full model')
legend('linear state space', 'ode4')

%% OUTPUT

% print model to the output file
save(fullfile(outputFolder, model_name), 'full_wec_ss')


