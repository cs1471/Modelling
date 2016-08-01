function [cv,firingRateHat,varargout]=runExperim(T,distribution,numSpikeTrains,param,theoDist)
% input:
%       T: length of spike train in seconds
%       distribution: 'exponential', 'gamma', or 'inversegaussian'
%       numSpikeTrains:  the number of spike trains to be generated
%       param: the scalar or vector that defines the parameters of the
%               distribution to be used
%       theoDist:  the name of the theoretical distribution we want to
%               compare the pooled spike trains -- set it to 'exponential'
% output:
%       cv: coefficient of variation of pooled ISIs
%       firingRateHat: 
%
%
%----
% Example:
% T = 1; % second
% distribution='gamma';
% numSpikeTrains =30;   
%     b = 60; 
%     a = 3; 
% param = [a b]
% theoDist = 'exponential' 
%
% [cv,firingRateHat,summary]=runExperim(T,distribution,numSpikeTrains,param,theoDist);
% printCellString(summary)  % if you want to print the summary of the
% results
%----
% Castellanos. January, 2008



spikeTimes = cell(numSpikeTrains,1);
isis =  cell(numSpikeTrains,1);
%spikeCounts = nan(numSpikeTrains,T/binSize); %,spikeCounts(idx,:)

for idx=1:numSpikeTrains
    [spikeTimes{idx},isis{idx}]= generateSpikeTrain(T,distribution,param);
end

allIsis = [isis{:}];

allSpikeTimes = [spikeTimes{:}];
allSpikeTimes = allSpikeTimes(allSpikeTimes<T); % focusing on time period of interest
sortAllSpikes=sort(allSpikeTimes);
isisSorted= diff(sortAllSpikes);

numBins = 100;
figure
set(gcf,'position',[360   279   637   643]);
subplot(2,1,1)
hist(isisSorted,100)
set(gca,'fontsize',13);
title(['ISISs histogram. ',distribution, '; ', num2str(numSpikeTrains),' spike trains. numBins = ',num2str(100)],'fontsize',14)

subplot(2,1,2)
myData = isisSorted;
ppPlot(myData,theoDist)

set(gca,'fontsize',13);
% obtain isis CV
cv = std(isisSorted)/mean(isisSorted)

% obtain estimated firing rate
firingRateHat = length(allSpikeTimes)/(T*numSpikeTrains);

% Write down summary of experiment:
summary{1}= strcat('ISIs~',distribution,' - param : [ ',num2str(param),']');
summary{2}=strcat('T = ',num2str(T),' seconds;  Num spike trains: ', num2str(numSpikeTrains));
summary{3}=strcat('firingRateHat = ',num2str(firingRateHat),' spike/seconds;  cv = ',num2str(cv));

varargout(1) = {summary};


end