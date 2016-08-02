% [NEWSIG, EVECS, EVALS, MAT] = pca(SIG,WHITEN)
%
% If SIG is a matrix whose columns are sample vectors drawn from a  
% probability density, NEWSIG is matrix whose columns have been
% projected onto the principal components of the data.  If 
% WHITEN is non-zero (default), the values will be scaled to unit
% variance (i.e: the covariance matrix of NEWSIG will be the identity 
% matrix.
% 
% EVECS and EVALS are the eigenvectors and eigenvalues of the original
% covariance matrix (i.e., SIG * SIG' = EVECS * EVALS * EVECS').
% 
% MAT contains the transformation matrix:
%   NEWSIG = MAT * SIG

% Robert Buccigrossi, Spring 1997.
% Modified by EPS, 10/97, to return eigenvector instead of whitening matrix

function [NEWSIG, EVECS, EVALS, MAT] = pca(SIG,WHITEN)

if (exist('WHITEN') ~= 1)
  WHITEN = 1;
end

numsigs = size(SIG,2);
[V,D] = eig(innerProd(SIG')/numsigs);

% Reverse sort
D = diag(D);
[junk,Ind] = sort(D);
D = D(Ind(size(Ind,1):-1:1));
V = V(:,Ind(size(Ind,1):-1:1));

EVALS = D;
EVECS = V;

if WHITEN
  nonZeroInd = find(abs(D) > eps);
  whitener = diag(sqrt(1./D(nonZeroInd))');
  MAT = whitener*EVECS';
else
  MAT = EVECS';
end

NEWSIG = MAT*SIG; 
