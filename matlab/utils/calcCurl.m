% curlV = calcCurl(V)
%
% Calculates the curl of a vector field with the vectors in the first
% dimension and the spatial information in the higher dimensions.
%
% VFt: Generally a 4D array, with the first dimension of length 3 being the
%      vector dimension. Dimensions 2 to 4 are the spatial dimensions.
% samplePitch: The distance between sample points in each dimension.
%              (default all 1)
%
function curlV = calcCurl(V, samplePitch)
  nbOutputDims = 3;
  if nargin < 2
    samplePitch = ones(1, nbOutputDims);
  end
  
  VFt = fftnButDim1(V); % Fourier transform
  curlVFt = calcCurlFt(VFt, samplePitch); % Calculate the curl on the Fourier transform of the field
  curlV = ifftnButDim1(curlVFt); % and back to the spatial domain
end

% N-1 dimensional Fast Fourier Transform on all but the first dimension
function result = fftnButDim1(F)
    result = zeros(size(F), class(F));
    rangeI = arrayfun(@(len) (1:len), size(F), 'UniformOutput',false);
    rangeI = rangeI(2:end);
    for inDimIdx = 1:size(F,1)
        result(inDimIdx,rangeI{:}) = fftn(F(inDimIdx,rangeI{:})); % First dimension is singleton, ignored by fftn
    end
end
% N-1 dimensional Inverse Fast Fourier Transform on all but the first dimension
function result = ifftnButDim1(FFt)
    result = zeros(size(FFt), class(FFt));

    fRangeI = arrayfun(@(len) (1:len), size(FFt), 'UniformOutput',false);
    fRangeI = fRangeI(2:end);
    for outDimIdx = 1:size(FFt,1)
        result(outDimIdx,fRangeI{:}) = ifftn(FFt(outDimIdx,fRangeI{:})); % Move the Fourier transform outside of the 'dot' product
    end
end