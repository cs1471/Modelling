%question 1a
[spikeTime, ISI, spikeCounts] = generateSpikeTrain(30,'exponential',20);
cv = std(ISI)/mean(ISI);
figure(1);
subplot(1,2,1)
hist(ISI);
subplot(1,2,2)
ppPlot(ISI,'exponential')

%%
%question 1b
nSpike = 30;
spikeTime1 = [];
spikeTimes = [];
for i = 1:nSpike
    spikeTime1 = generateSpikeTrain(1,'exponential',20);
    spikeTimes = [spikeTimes, spikeTime1];
end

spikeTimes_sort = sort(spikeTimes);
ISI_pool = diff(spikeTimes_sort);
cv_pool = std(ISI_pool)/mean(ISI_pool);
figure(2);
subplot(1,2,1)
hist(ISI_pool);
subplot(1,2,2)
ppPlot(ISI_pool,'exponential')

%%
clear
%question 2a
[spikeTime, ISI, spikeCounts] = generateSpikeTrain(30,'gamma',[1 20]);
cv = std(ISI)/mean(ISI);
figure(3);
subplot(1,2,1)
hist(ISI);
subplot(1,2,2)
ppPlot(ISI,'exponential')

%%
%question 2b
clear
nSpike = 30;
spikeTime1 = [];
spikeTimes = [];
for i = 1:nSpike
    spikeTime1 = generateSpikeTrain(1,'gamma',[1 20]);
    spikeTimes = [spikeTimes, spikeTime1];
end

spikeTimes_sort = sort(spikeTimes);
ISI_pool = diff(spikeTimes_sort);
cv_pool = std(ISI_pool)/mean(ISI_pool);
figure(4);
subplot(1,2,1)
hist(ISI_pool);
subplot(1,2,2)
ppPlot(ISI_pool,'exponential')

%%
clear
%question 3a
[spikeTime, ISI, spikeCounts] = generateSpikeTrain(30,'gamma',[3 60]);
cv = std(ISI)/mean(ISI);
figure(5);
subplot(1,2,1)
hist(ISI);
subplot(1,2,2)
ppPlot(ISI,'exponential')

%%
%question 3b
clear
nSpike = 30;
spikeTime1 = [];
spikeTimes = [];
for i = 1:nSpike
    spikeTime1 = generateSpikeTrain(1,'gamma',[3 60]);
    spikeTimes = [spikeTimes, spikeTime1];
end

spikeTimes_sort = sort(spikeTimes);
ISI_pool = diff(spikeTimes_sort);
cv_pool = std(ISI_pool)/mean(ISI_pool);
figure(6);
subplot(1,2,1)
hist(ISI_pool);
subplot(1,2,2)
ppPlot(ISI_pool,'exponential')

%%
clear
%question 4a
[spikeTime, ISI, spikeCounts] = generateSpikeTrain(30,'inversegaussian',[20 1]);
cv = std(ISI)/mean(ISI);
figure(7);
subplot(1,2,1)
hist(ISI);
subplot(1,2,2)
ppPlot(ISI,'exponential')

%%
%question 3b
clear
nSpike = 30;
spikeTime1 = [];
spikeTimes = [];
for i = 1:nSpike
    spikeTime1 = generateSpikeTrain(1,'inversegaussian',[20 1]);
    spikeTimes = [spikeTimes, spikeTime1];
end

spikeTimes_sort = sort(spikeTimes);
ISI_pool = diff(spikeTimes_sort);
cv_pool = std(ISI_pool)/mean(ISI_pool);
figure(8);
subplot(1,2,1)
hist(ISI_pool);
subplot(1,2,2)
ppPlot(ISI_pool,'exponential')
