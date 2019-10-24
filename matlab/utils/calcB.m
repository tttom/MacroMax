% B = calcB(E, samplePitch, wavenumber)
%
% Calculates the magnetic flux density B
%
function B = calcB(E, samplePitch, wavenumber)
  if nargin < 3
    wavenumber = 1.0;
  end
  
  B = (1i/wavenumber) * calcCurl(E, samplePitch);
end