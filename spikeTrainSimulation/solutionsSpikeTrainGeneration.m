%% Exponential

clear

theoDist = 'exponential'

distribution='exponential';
numSpikeTrains = 30; %1; %
T =  1;% 30; % 

param = 20; %  =  firingRate 

[cv,firingRateHat,summary]=runExperim(T,distribution,numSpikeTrains,param,theoDist);
printCellString(summary)



%% Gamma
clear

theoDist = 'exponential'

distribution='gamma';
numSpikeTrains =30;%  1;%  
T =1; %  30;%   

    invB = 60; % 20; %
    a = 3; %  1; % 
param = [a invB]


[cv,firingRateHat,summary]=runExperim(T,distribution,numSpikeTrains,param,theoDist);
printCellString(summary) 


%%
clear

theoDist = 'exponential'


binSize = .001;
distribution='inversegaussian';

numSpikeTrains =  30; %1; % 
T =  1; % 30; %

firingRate = 20;
lambda =100; % 1; % 
param = [firingRate lambda]

[cv,firingRateHat,summary]=runExperim(T,distribution,numSpikeTrains,param,theoDist);
printCellString(summary)