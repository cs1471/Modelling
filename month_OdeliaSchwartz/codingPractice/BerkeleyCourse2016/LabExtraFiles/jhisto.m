% [N, Y, X] = JHISTO(YIMAGE, XIMAGE, NBINS_OR_BINSIZE, BIN_CENTER)
%
% Compute joint histogram of Yimage and Ximage.
%
% NBINS_OR_BINSIZE (optional, default = 101) specifies either
% the number of histogram bins, or the negative of the binsize.
% It can be a [Y,X] 2-vector or it can be a scalar.
%
% BIN_CENTER (optional, default = mean2(MTX)) specifies a center position
% for (any one of) the histogram bins.
% It can be a [Y,X] 2-vector or it can be a scalar.

% EPS, Spring 97.

function [N, Y, X] =  jhisto(Yim, Xim, nbins, binctr)

%% NOTE: THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD)

%fprintf(1,'WARNING: You should compile the MEX code for "jhisto".  It ...
    %is MUCH faster.  Functionality may be slightly different.\n');

if (exist('nbins') == 1) 
  nbins = abs(round(nbins));
  if (prod(size(nbins)) == 1)
    nbins = [nbins,nbins];
  else
    nbins = nbins(:);
  end
else
  nbins = [101, 101]';
end

if (exist('binctr') ~= 1)
  binctr = [mean2(Yim); mean2(Xim)];
end

nbins=nbins-1;
[Xmn, Xmx] = range2(Xim);
X = Xmn + (Xmx-Xmn)*[0:nbins(2)]/nbins(2);
[Ymn, Ymx] = range2(Yim);
Y = Ymn + (Ymx-Ymn)*[0:nbins(1)]/nbins(1);
Ybs = (Y(2)-Y(1))/2;
nbins=nbins+1;
N = zeros(nbins);
for yind=1:nbins(1)
  ind1 = find(Yim > (Y(yind)-Ybs));
  ind2 = find(Yim(ind1) < (Y(yind)+Ybs));
  ind = ind1(ind2);
  if(size(ind,2)>0)
    N(yind,:) = hist(Xim(ind),X);
  end
end


