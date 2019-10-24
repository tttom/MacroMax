% curlVFt = calcCurlFt(VFt)
%
% Calculates the curl of a vector field with the vectors in the first
% dimension and the spatial information Fourier transformed in both the
% input and the output.
%
% VFt: Generally a 4D array, with the first dimension of length 3 being the
%      vector dimension. Dimensions 2 to 4 are Fourier-transformed with the DC
%      component in the first element.
% samplePitch: The distance between sample points in each dimension.
%              (default all 1)
%
function curlVFt = calcCurlFt(VFt, samplePitch)
  nbOutputDims = 3;
  if nargin < 2
    samplePitch = ones(1, nbOutputDims);
  end
  
  dataSize = size(VFt);
  nbInputDims = dataSize(1);
  if nbInputDims <= 1
    error('The vector field should have as first dimension 2 or 3, not %d.', nbInputDims);
  end
  dataSize = dataSize(2:end);
  dataSize(end+1:nbOutputDims) = 1; % Pad with 1s
  samplePitch(end+1:nbOutputDims) = 1; % Pad with 1s so that kRange [0]s are added

  kRanges = arrayfun(@(sp,rl) 2*pi/(sp*rl) * ifftshift([1:rl]-1-floor(rl/2)),...
    samplePitch, dataSize, ...
    'UniformOutput',false);
    
  kRangeI = arrayfun(@(len) (1:len), dataSize, 'UniformOutput',false);
  
  if size(VFt, 1) < nbOutputDims
    VFt(nbOutputDims, 1) = 0; % set  missing data to 0
  end
  curlVFt = VFt; % Initialized array of the same size
  for dimIdx = 1:nbOutputDims
    otherDims = 1 + mod(dimIdx + [1 2] - 1, nbOutputDims);
    curlVFt(dimIdx, kRangeI{:}) = bsxfun(@times,...
        shiftdim(1i*kRanges{otherDims(1)}, 1-otherDims(1)),...
        VFt(otherDims(2), kRangeI{:})...
      ) - bsxfun(@times,...
        shiftdim(1i*kRanges{otherDims(2)}, 1-otherDims(2)),...
        VFt(otherDims(1), kRangeI{:})...
      );
  end
end