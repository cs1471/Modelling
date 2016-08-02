
function theres = gaussEstim(l, lself, n, a);
% function theres = gaussEstim(l, lself, n, a);
% estimate Gaussian component of the GSM
% a Rayleigh parameter
% n number of filters
% lself is filter response; l is gain pool response

  thefac = (1./(a.*lself.^2).^.25) .* (l.^2./lself.^2).^.25;
  thefac=1./thefac;
  theres = sign(lself).*thefac.* ...
  ((besselk(.5*(-n+a), ...
   l*sqrt(a))./besselk(.5*(-n+1+a),l*sqrt(a))));

