% -------------------------------------------------------------------------
% Name:            WAMIT2json.m
% Description:     Take wamit outfiles and process hydrodynamic coef and irf data, build the radiation state space, and save all in a json file
%
% Author:          Antoine Dupuis
% Collaborator:    Vinicius 
% Date created:    January 2025
% Data:            - 
% Project:         - 
%
% Inputs: 
% - WAMIT model name: BEM data
%
% Outputs:
% - json file: hydro data, irf, and radiation state space
%
% Dependencies:
% - radiation2ss.m: function to compute the radiation state space subsystem
% -------------------------------------------------------------------------

clear
clc

%% INPUT

filePath = pwd + "\data\bem\wamit\sphere\";
fileName = "xModel";

rho = 1025;

n_state = 6;

plot_flag = 0;

%% SET UP SCRIPT

% make local paths
myfodler = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(myfodler, 'source')));

%% FREQUENCY DOMAIN

% Add mass and Rad res
Amass33Inf              = zeros(1,1);
Amass33_0               = zeros(1,1); 

copyfile([filePath + fileName + ".1"], [filePath + fileName + "Out.txt"]) % change wamit file to .txt

% fix rad. res. for 0 and inf frequency
fileID                  = fopen([filePath + fileName + 'Out.txt'],'r');

i = 0;          

line = fgetl(fileID);                                           % Skip the header lines
while i ~= -1
    % Read the next line
    line                = fgetl(fileID);
    i = i+1;               
    fields              = strsplit(line);                       % Split the line into fields based on whitespace
    values              = str2double(fields);
    if values(2) == -1                    
        Amass33_0(1,1) = values(5);                            % Store added mass for w_inf
    elseif values(2) == 0                    
        Amass33Inf(1,1)  = values(5);                            % Store added mass for w_0
    elseif values(2) > 0
        i = -1;       
    end
end

fclose(fileID);

M                       = readmatrix([filePath + fileName + 'Out.txt']);
Amass33                 = M(:,4); 
radRes33                = M(:,5);     

% Excitation coef
copyfile([filePath + fileName + '.3'] , [filePath + fileName + 'Out.txt']) % change wamit file to .txt

M                       = readmatrix([filePath + fileName + 'Out.txt']);
exc33                   = M(:,6) + 1i*M(:,7); 

delete([filePath + fileName + 'Out.txt'])                       % delete .txt file

w                       = M(:,1);                               % angular frequency

% Dimensionalize
g = 9.81;
Amass33                 = Amass33.*rho.*1^3;
Amass33_0               = Amass33_0.*rho.*1^3;
Amass33Inf              = Amass33Inf.*rho.*1^3;
radRes33                = radRes33.*rho.*w.*1^3;
exc33                   = exc33.*rho.*g;

%% TIME DOMAIN

% diffIRF JR.3
copyfile([filePath + '\' + fileName + '_JR.3'] , [filePath + '\' + fileName + 'Out.txt']) % change wamit file to .txt
M         = readmatrix([filePath + '\' + fileName + 'Out.txt']);
dIRF33    = M(:,2); 

% time
dIRFtime  = M(:,1);

% radIRF KR.1
copyfile([filePath + '\' + fileName + '_KR.1'] , [filePath + '\' + fileName + 'Out.txt']) % change wamit file to .txt
M         = readmatrix([filePath + '\' + fileName + 'Out.txt']);
rIRF33    = M(:,2);

delete([filePath + '\' + fileName + 'Out.txt'])           

% time
rIRFtime  = M(:,1);

% Dimensionalize
rIRF33 = rIRF33.*rho.*1^3;
dIRF33 = dIRF33.*rho.*g*1^2; 

%% RADIATION SS

% compute radiation state space
rad_ss = radiation2ss(n_state, w, Amass33, radRes33, Amass33Inf, 'plot');

% get radiation state space matrices
Ar = rad_ss.A;

Br = rad_ss.B;

Cr = rad_ss.C;

Dr = rad_ss.D;

% store in structure
rad_m = struct('Ar', Ar, 'Br', Br, 'Cr', Cr, 'Dr', Dr);

%% CHECK

if plot_flag == 1

    % added mass
    figure
    plot(w, Amass33, 'LineWidth', 1.5); hold on
    yline(Amass33_0, '--', 'A_{33}(0)');
    yline(Amass33Inf, '--', 'A_{33}(\infty)');
    grid on
    xlabel('\omega [rad/s]')
    ylabel('Added Mass A_{33}')
    title('Heave Added Mass vs Frequency')
    legend('A_{33}(\omega)','A_{33}(0)','A_{33}(\infty)','Location','best')
    
    % radiation damping
    figure
    plot(w, radRes33, 'LineWidth', 1.5)
    grid on
    xlabel('\omega [rad/s]')
    ylabel('Radiation Damping B_{33}')
    title('Heave Radiation Damping vs Frequency')
    
    % excitation coef
    figure
    subplot(2,1,1)
    plot(w, abs(exc33), 'LineWidth', 1.5)
    grid on
    xlabel('\omega [rad/s]')
    ylabel('|F_{exc,33}|')
    title('Excitation Force Magnitude (Heave)')
    
    % excitation phase
    subplot(2,1,2)
    plot(w, angle(exc33)*180/pi, 'LineWidth', 1.5)
    grid on
    xlabel('\omega [rad/s]')
    ylabel('Phase [deg]')
    title('Excitation Force Phase (Heave)')
    
    % radiation irf
    figure
    plot(rIRFtime, rIRF33, 'LineWidth', 1.5)
    grid on
    xlabel('Time [s]')
    ylabel('K_{33}(t)')
    title('Radiation Impulse Response Function (Heave)')
    
    % excitation irf
    figure
    plot(dIRFtime, dIRF33, 'LineWidth', 1.5)
    grid on
    xlabel('Time [s]')
    ylabel('J_{33}(t)')
    title('Diffraction Impulse Response Function (Heave)')

end

%% SAVE OUTPUT

jsonFile = fullfile(filePath, [fileName + ".json"]);

matlab2json(jsonFile, ...
    'w', w, ...
    'Fe_mod', abs(exc33), ...
    'Fe_ang', angle(exc33), ...
    'B', radRes33, ...
    'ma', Amass33, ...
    'ma_inf', Amass33Inf, ...
    'rad_m', rad_m, ...
    'fe_irf', dIRF33, ...
    't_irf', dIRFtime);



