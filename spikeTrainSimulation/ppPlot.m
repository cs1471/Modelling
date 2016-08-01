function ppPlot(myData,theoDist)
% ppPlot
%       inputs:  
%            myData: vector of observations 
%            theoDist:  theoretical distribution against which we want 
%                       to compare the observations
%
%
% Example:
% myData = isis;
% distribu = 'exponential'
% ppPlot(myData,theoDist);
%--------------
% By: Castellanos, January 2008

% obtain the empirical distribution:
[eF,x]=ecdf(myData);
% eF = vector of values of the empirical cdf evaluated at:
%  x

% obtain the theoretical distribution
switch theoDist
    case 'exponential'
        muHat = mean(myData);
        tF = expcdf(x, muHat);
    otherwise
        disp(strcat('Warning: theoDist = ',theoDist, 'is not programmed. Implement it yourself: modify ppPlot.m' ))
end


hold on
% plot the empirical vs the theoretical probabilities
plot(eF,tF,'.b','markersize',5);
% plot a line of reference
plot([0 1], [0 1],'r','linewidth',0.5);

title(strcat('pp-plot against: ',theoDist, '  distribution.'),'fontsize',14);



end