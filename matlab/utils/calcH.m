% H = calcH(E, samplePitch, wavenumber, mu, zeta)
%
% Calculates the magnetizing field H
%
function H = calcH(E, samplePitch, wavenumber, mu, zeta)
  if nargin < 3
    wavenumber = 1.0;
  end
  if nargin < 4
    mu = 1.0;
  end
  if nargin < 5
    zeta = 0.0;
  end
  
  B = calcB(E, samplePitch, wavenumber);
  B = permute(B, [1 5 2 3 4]);
  pE = permute(E, [1 5 2 3 4]);
  mu0muH = B - (arrayMatMul(zeta, pE) ./ Const.c);
  clear pE;
  H = arrayMatDivide(mu, mu0muH ./ Const.mu0);
  H = ipermute(H, [1 5 2 3 4]);
end