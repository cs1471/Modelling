% Matlab code to reproduce Figure 2C and 2G in
% Shadlen, MN and Newsome, WT (1998). 
% The variable discharge of cortical neurons: 
% implications for connectivity, computation and information coding. 
% Journal of Neuroscience 18: 3870-3896.
%----------------
% Zhanwu Liu, 2011

%% For Figure 2C, excitatory input only 
% Simulate the input neurons
input_rate = 50; % mean rate 50 spikes/neuron
input_number = 300; %300 neurons  -- note: 300 input generate 6-7 spikes in the output neuron, while 270 input generate ~5 spikes
simulation_time = 100; %100 ms 

parameter_exponential = 1000/input_rate; %mean ISI

spike_ISIs = exprnd(parameter_exponential, input_number, 20); % I simulate 20 ISIs to make sure there is enough, expected value is 5

spike_times = cumsum(spike_ISIs, 2); 


% change to 0/1 representation, and only take the 0-100 ms
mat_input = zeros(input_number,simulation_time);
for  i = 1:input_number
    ind_spike = ceil(spike_times(i,:));
    ind_spike = ind_spike(ind_spike<=simulation_time);
    mat_input(i,ind_spike)=1;
end

% The count is used as the real input
num_input = sum(mat_input);

% exponential decay parameters
tao = 20;
decay_ratio = exp(-1/tao); %this is the ratio of decay after each ms

%threshold
threshold = 150; %150 is the number of steps as defined in the paper


%Simulate the output neurons, record the height, and spike times
height = zeros(simulation_time,1);
height(1) = num_input(1);
spikes = [];
for i = 2:simulation_time %loop over time
    height(i) = height(i-1)*decay_ratio + num_input(i);
    if height(i) > threshold  
        spikes = [spikes, i];
        height(i) = 0; %reset to zero
    end
end


% convert the height into voltage, and plot 
% (height 150 <-> -55 mV, height 0 <-> -70 mV)
% action potential voltage maximum 10 mV
voltage = height/10-70;
voltage(spikes) = 10;
plot(voltage);
ylim([-75 15])
title('Counting model: {\bf excitatory} input','fontsize',12)
set(gca,'fontsize',13);


%% For Figure 2G, both excitatory and inhibitory input

input_rate = 50; % mean rate 50 spikes/neuron
input_number = 300; %300 neurons
simulation_time = 100; %100 ms 

parameter_exponential = 1000/input_rate; %mean ISI

% Simulate the excitatory neurons
spike_ISIs_e = exprnd(parameter_exponential, input_number, 20); % I simulate 20 ISIs to make sure there is enough, expected value is 5
spike_times_e = cumsum(spike_ISIs_e, 2); 

% change to 0/1 representation, and only take the 0-100 ms
mat_input_e = zeros(input_number,simulation_time);
for  i = 1:input_number
    ind_spike = ceil(spike_times_e(i,:));
    ind_spike = ind_spike(ind_spike<=simulation_time);
    mat_input_e(i,ind_spike)=1;
end

% The count is used as the real input
num_input_e = sum(mat_input_e);

% Simulate the inhibitory neurons
spike_ISIs_i = exprnd(parameter_exponential, input_number, 20); % I simulate 20 ISIs to make sure there is enough, expected value is 5
spike_times_i = cumsum(spike_ISIs_i, 2); 

% change to 0/1 representation, and only take the 0-100 ms
mat_input_i = zeros(input_number,simulation_time);
for  i = 1:input_number
    ind_spike = ceil(spike_times_i(i,:));
    ind_spike = ind_spike(ind_spike<=simulation_time);
    mat_input_i(i,ind_spike)=1;
end

% The count is used as the real input
num_input_i = sum(mat_input_i);

% exponential decay parameters
tao = 20;
decay_ratio = exp(-1/tao); %this is the ratio of decay after each ms

%threshold
threshold2 = 15; %15 is the number of steps as defined in the paper

%Simulate the output neurons with ex and in neurons
height_2 = zeros(simulation_time,1);
height_2(1) = max(num_input_e(1)-num_input_i(1),0);
spikes_2 = [];
for i = 2:simulation_time %loop over time
    height_2(i) = height_2(i-1)*decay_ratio + num_input_e(i)-num_input_i(i);
    if height_2(i) > threshold2
        spikes_2 = [spikes_2, i];
        height_2(i) = 0;
    end
    if height_2(i) < 0
        height_2(i) = 0; %lower bound
    end
end
voltage_2 = height_2-70;
voltage_2(spikes_2) = 10;
plot(voltage_2);
ylim([-75 15])
title('Counting model: {\bf balanced} input','fontsize',12)
set(gca,'fontsize',13);