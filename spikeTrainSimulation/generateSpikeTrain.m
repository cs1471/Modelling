function [spikeTimes varargout]= generateSpikeTrain(T,distribution,param,varargin)
% generateSpikeTrain
%       inputs:  
%            T: length of spike train (in seconds)
%            distribution:  string specifying the distribution of the 
%                           interspike intervals. In our case consider:
%                           'exponential'
%                           'inversegaussian'
%            param: scalar or vector that defines the distribution:
%                   for 'exponential'  param is a scalar specifying the firingRate                                     
%                   for 'inversegaussian' param is a vector specifying [mu, lambda]
%                   for 'gamma' param is a vector specifying [a b] 
%                            a = shape, b = scale.  
%                            When a = 1 the gamma
%                            is an exponential function and b = firingRate
%                            in our case.
%                   two parameters:  [firingRate lambda]
%            binSize (optional): bin size (in seconds) for calculating  
%                                the spike counts in interval (0,T)
%                                If not specified, bin size is 50mS.
%       outputs:
%            spikeTimes:  the generated spike train
%            isis:        (optional) the ISIs 
%            spikeCounts: (optional) the spikeTimes binned in intervals of
%                         size binSize over interval (0,T)
%
%
% Example:
% T = 1; % second
% distribution = 'exponential';
% param = 50;       % 50 spikes per second = firing rate
% binSize = 0.001;  % 1 mili second 
% [spikeTimes,isis,spikeCounts]= generateSpikeTrain(T,distribution,param,binSize);
%--------------
% By: Castellanos, January 2008

if nargin == 4
    binSize = varargin{1}; 
else
    binSize = 0.05; % 50mS (= 0.05 seconds)
end

switch distribution
    case 'exponential'
        param = 1/param; % the user inputs firing rate, we modify the parameter
                         % to match the parametrization used in matlab
    case 'inversegaussian'
        firingRate = param(1);
        lambda = param(2);
        
        mu = 1/firingRate; % the user inputs firing rate, we modify the parameter
                           % to match the parametrization used in matlab
        param = [mu lambda];
    case 'gamma'
        a = param(1); % shape. When a=1, we have an exponential.
        
        b = 1/param(2);  % scale parameter.  the user inputs 1/mu, we modify the parameter
                         % to match the parametrization used in matlab
        param = [a b];
    otherwise
        disp(strcat('Warning!!  Your choice of distribution: ',distribution,'  - is not considered in this function.'))
end


% defining probability distribution
pd = ProbDistUnivParam(distribution,param);


% initializing variables
spikeTimesCell = {0};
isis = [];
idx=0;

% generate ISIs (and spikeTimes)
while spikeTimesCell{end}<T
    % taking care of indexes
    idx=idx+1;

    % generate a random sample from the defined distribution
    oneISI = random(pd,1);
    
    % store generated ISI
    isis = [isis, oneISI];
    
    % record spike times accordingly
    if idx==1
        spikeTimesCell{1} = oneISI;
    else
        spikeTimesCell{idx} = spikeTimesCell{idx-1}+oneISI; 
    end
   
end

spikeTimes =[spikeTimesCell{:}];
spikeTimes = spikeTimes(spikeTimes<T); % focusing on time interval of interest

% obtain spikeCounts
%[spikeCounts toThrow] = hist(spikeTimes, T/binSize);
spikeCounts = histc(spikeTimes,0:binSize:T);
spikeCounts = spikeCounts(1:(end-1));


if isempty(spikeCounts)
     spikeCounts = zeros(1,T/binSize);
 end

varargout={isis,spikeCounts};

end