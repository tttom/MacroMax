% [varargout] = calcRanges(rangeLengths, samplePitches, centerOffsets)
%
% returns uniformely spaced ranges of length rangeLengths(idx) with a elements spaced by
% samplePitches(idx) and centered on centerOffsets(idx). The center element
% is defined as the one in the center for an odd number of elements and the
% next one for an even number of elements. If a scalar is specified as sample pitch
% or center offset, it is used for all ranges. The default sample pitch is 1
% and the default center offset is 0.
%
% Example:
%    [xRange, yRange] = calcRanges([128 128], [1 1]*1e-6);
%
function [varargout] = calcRanges(rangeLengths, samplePitches, centerOffsets)
    nbDims = nargout;
    if nargin<2 || isempty(samplePitches)
        samplePitches = ones(1,nbDims);
    end
    if nargin<3 || isempty(centerOffsets)
        centerOffsets = zeros(1,nbDims);
    end
    % Make sure the vectors are of the same length
    rangeLengths(end+1:nbDims) = 1;
    samplePitches(end+1:nbDims) = samplePitches(end);
    centerOffsets(end+1:nbDims) = centerOffsets(end);
    
    varargout = arrayfun(@(co,sp,rl) co + sp*([1:rl]-1-floor(rl/2)),...
        centerOffsets(1:nbDims),samplePitches(1:nbDims),rangeLengths(1:nbDims),...
        'UniformOutput',false);
end