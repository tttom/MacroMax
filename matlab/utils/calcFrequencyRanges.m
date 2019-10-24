% varargout=calcFrequencyRanges(varargin)
%
% Determine equivalent frequency ranges for given time ranges.
% The results are ifftshifted so that the zero frequency is in the first
% vector position, unless the final input arguments is true or is a string
% that starts with 'center'.
% Obsolete: If matrices are specified as input arguments, these are assumed to be plaid as provided by ndgrid.
% All should have the same dimensions and the number of non-singleton dimensions should equal the
% number of matrix inputs.
%
% Examples:
%    [xfRange,yfRange]=calcFrequencyRanges(xRange,yRange);
%    [xfRange,yfRange]=calcFrequencyRanges(xRange,yRange,'centered');
%
function varargout = calcFrequencyRanges(varargin)
    if nargin>nargout && ((isscalar(varargin{end}) && islogical(varargin{end})) || (ischar(varargin{end}) && numel(varargin{end}>=6)))
        centered = (isscalar(varargin{end}) && varargin{end}==true) || (ischar(varargin{end}) && strcmpi(varargin{end}(1:6),'center'));
        ranges = varargin(1:end-1);
    else
        centered = false;
        ranges = varargin;
    end
    % empty inputs? => any(cellfun(@(e) any(size(e)==0),ranges))
%     matrixInputs=all(cellfun(@(e) sum(size(e)>1),ranges)==numel(ranges));
    
    varargout = cell(1,nargout);
    for rngIdx = 1:max(1,nargout)
        rng = ranges{rngIdx};
%         if matrixInputs,
%             rng=shiftdim(rng,rngIdx-1);
%             rng=rng(:,1);
%         end
        nb = numel(rng);
        dt = (rng(end)-rng(1))/(nb-1);
        if dt ~= 0
            fRange = ([1:nb]-1-floor(nb./2))/(nb*dt);
        else
            fRange = 0.0*rng;
        end
        if ~centered
            fRange = ifftshift(fRange);
        end
        varargout{rngIdx} = fRange;
    end
end
