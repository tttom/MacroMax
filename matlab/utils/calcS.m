%
% S = calcS(E, samplePitch, wavenumber, mu, zeta)
%
% Calculates the Poynting vector based on the electric field and the material properties.
%
function S = calcS(E, samplePitch, wavenumber, mu, zeta)
  if nargin < 3
    wavenumber = 1;
  end
  if nargin < 4
    mu = 1.0;
  end
  if nargin < 5
    zeta = 0.0;
  end
  
  H = calcH(E, samplePitch, wavenumber, mu, zeta);
  S = 0.5 * real(cross(E, conj(H))); % Time averaged Poynting vector
end