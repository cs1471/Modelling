function [LL, PP, like, YY] = emfa(XX, KK, varargin)
% emfa - maximum likelihood factor analysis using EM: [L, P, like, Y] = emfa(X, K)
%

niter = 1000;
tol = 1e-7;
initial = [];
assignopts(who, varargin);

[DD,NN] = size(XX);

XX=bsxfun(@minus, XX, mean(XX, 2));
XX2=XX*XX';
diagXX2=diag(XX2);

if (isempty(initial))
  cX = cov(XX');
  scale = det(cX)^(1/DD);
  if scale < eps scale = 1; end
  LL = randn(DD,KK)*sqrt(scale/KK);
  PP = diag(cX);
else
  LL = initial.loadings;
  PP = initial.uniquenesses;
end


II = eye(KK);

if (nargout > 2)
  like = zeros(1, niter);
end

for iter = 1:niter
  
  PPinv = diag(1./PP);
  PPinvLL = bsxfun(@rdivide,LL, PP);
  
  YYcov = inv(LL'*PPinvLL + II);
  YY = YYcov*PPinvLL'*XX;
  YY2 = NN*YYcov + YY*YY';

  if nargout > 2 | tol > 0
    XXprec = PPinv - PPinvLL*YYcov*PPinvLL';
    like(iter) = 0.5*NN*(log(det(XXprec))) - 0.5*sum(sum(XXprec.*XX2));
  end  

  LL = XX*YY'/YY2;
  PP = (diagXX2 - sum((LL*YY).*XX, 2))/NN;

  if tol > 0 & iter > 1
    if abs(diff(like(iter-1:iter))) < tol*diff(like([1,iter]))
      like = like(1:iter);
      break;
    end
  end
end

