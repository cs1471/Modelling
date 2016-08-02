% [NEWSIG, MAT, EVALS] = ica4(SIG)
%
% If SIG is a matrix of col vector signals, NEWSIG are the signals
% separated out into independent components through Cardoso's Fourth
% Order Blind Identification (FOBI) algorithm.  
% MAT gives the transformation matrix:
%     PSIG = MAT * SIG
%
%  EVALS gives the eigenvalues of the 4th-order matrix.

% Robert Buccigrossi/Eero Simoncelli, Spring 1997.

function [NEWSIG, MAT, EVALS] = ica4(SIG)

[PSIG, EVECS, EVALS, PMAT] = mypca(SIG);
numsigs = size(PSIG,1);
sz = size(PSIG,2);

% We want to calculate (y' . y) * (y * y')

dot = sqrt(sum(PSIG .* PSIG));
doty = zeros(size(PSIG));
for i = 1:numsigs 
  doty(i,:) = PSIG(i,:) .* dot;
end

[V,D] = eig(innerProd(doty')/sz);

%% sort from highest to lowest eigenval:
D = diag(D);
[junk,Ind] = sort(D);
EVALS = junk(Ind(size(Ind,1):-1:1));
V =     V(:,Ind(size(Ind,1):-1:1));

MAT = V' * PMAT;
NEWSIG = V' * PSIG;
