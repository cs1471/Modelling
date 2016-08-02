
%%% This is a tutorial and exercises for sound statistics
%%% and linear mixing.

%%% Make sure to add paths of all directories associated with
%%% the tutorial. Also, note that some functions used are
%%% similar to built in Matlab functions -- you can use these
%%% or your own versions.

%%% Read the comments and copy each line of code into matlab.
%%% Note that in some places the code is incomplete and you
%%% need to fill in the pieces...
%%% Type 'help <function_name>' in Matlab window for any
%%% function you would like more information on.
%%% Type 'which <function_name>' in Matlab window for the
%%% location of a file.

%%% Odelia Schwartz, Berkeley summer course, 2016.
%%% Code with Luis Gonzalo Sanchez Giraldo.

%%% Linear mixing of sounds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

%% Read speech sounds
[y1,fs] = audioread('spd.wav');
[y2,fs] = audioread('spc.wav');

%% Take numsamps samples of each sound
numsamps = 40000;
y1=y1(1:numsamps);
y2=y2(1:numsamps);
figure(1)
subplot(2,2,1)
scatter(y1,y2, 'r.')
tmp = max(max(abs([y1,y2])));
axis([-tmp tmp -tmp tmp]);
set(gca, 'DataAspectRatio', [1 1 1])
title('Original Sources')
xlabel('y1')
ylabel('y2')

%% Listen to the sounds
% We have put an if 0 so that you can control if
% to hear the sound. You can copy and paste the lines
% into the Matlab window to hear the sounds.
if 0
soundsc(y1,fs)
soundsc(y2,fs)
end

%% Linearly mix the sounds
mix1 = .6; 
mix2 = .4;
y3 = mix1*y1 + (1-mix1)*y2;
y4 = mix2*y1 + (1-mix2)*y2;
subplot(2,2,2)
scatter(y3,y4, 'b.' )
set(gca, 'DataAspectRatio', [1 1 1])
title('Observed mixtures')
xlabel('mix1')
ylabel('mix2')

%% Listen to the mixed sounds
if 0
soundsc(y3,fs)
soundsc(y4,fs)
end

sig = [y3, y4]';

%% TODO:
%% Try PCA on sig (eg, using mypca function). 
%% Does it work in separating the sounds? Why?
%% Plot a scatter plot of the PCA components as above
%% in the third subplot (eg, subplot(2,2,3).

%% Listen to the pca sounds
if 0
soundsc(pcasig(1,:),fs)
end

%% Try ICA. Does it work in separating the sounds? Why?
%% You can use the function ica4.
%% Plot a scatter plot of the ICA components as above
%% in the fourth subplot (eg, subplot(2,2,4).

%% Listen to the ica sounds
if 0
soundsc(icasig(1,:),fs)
soundsc(icasig(2,:),fs)
end

%% TODO.
%% Look at the statistics of the sounds using histo. What is the kurtosis?
%% Generate a Gaussian the same size as y1. How do the speech signals compare 
%% to a Gaussian?
%% You can also compute the kurtosis of the pca and ica signals.


