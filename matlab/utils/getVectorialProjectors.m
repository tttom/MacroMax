%
% projF = getVectorialProjectors(samplePitch)
%
% Returns functions that help split a vectorfield into its transverse and longitudinal components for a specific sample pitch:
% longitudinalProjector(x) => Pi_L(x)
% transversalProjector(x) => Pi_T(x)
% longitudinalProjector(xFt) => FT(Pi_L(IFT(x)))
% transversalProjector(xFt) => FT(Pi_L(IFT(x)))
% K2Functor(dataSize) => K2
% fftnButDim1(x) => FT(x)
% ifftnButDim1(x) => IFT(x)
%
function projF = getVectorialProjectors(samplePitch)
    projF = struct();
    projF.longitudinalProjectorFt = @(xFt) longitudinalProjectorFtFunction(xFt, samplePitch);
    projF.transversalProjectorFt = @(xFt) xFt - projF.longitudinalProjectorFt(xFt);
    
    projF.fftnButDim1 = @fftnButDim1Function;
    projF.ifftnButDim1 = @ifftnButDim1Function;
    projF.longitudinalProjector = @(x) projF.ifftnButDim1(projF.longitudinalProjectorFt(projF.fftnButDim1(x)));
    projF.transversalProjector = @(x) projF.ifftnButDim1(projF.transversalProjectorFt(projF.fftnButDim1(x)));
    
    projF.K2Functor = @(dataSize) getK2(dataSize, samplePitch);
    projF.KFunctor = @(dataSize) getK(dataSize, samplePitch);
    projF.PiLFtFunctor = @(dataSize) getPiLFt(dataSize, samplePitch);
    projF.PiTFtFunctor = @(dataSize) getPiTFt(dataSize, samplePitch);
end

function result = getPiLFt(dataSize, samplePitch)
    nbDimsIn = min([numel(dataSize), numel(samplePitch)]);
    nbDimsOut = 3; % Maintain field in higher dimensions than the sample

    getKRange = @(dimIdx) 2*pi*calcFrequencyRanges(calcRanges(dataSize(dimIdx), samplePitch(dimIdx)));

    result = zeros([nbDimsOut nbDimsOut dataSize], class(dataSize));
    ftRangeI = arrayfun(@(len) (1:len), dataSize, 'UniformOutput',false);
    for outDimIdx = 1:nbDimsIn  % Saving memory by not storing the complete tensor
        kSubRangeOut = getKRange(outDimIdx);
        for inDimIdx = 1:nbDimsIn
            kSubRangeIn = getKRange(inDimIdx);
            % Direction unit vectors of two selected components in N-dimensional space, the undefined value at 0 cancels out with the next term
            nbCopies = dataSize;
            nbCopies([inDimIdx, outDimIdx]) = 1;
            result(outDimIdx, inDimIdx, ftRangeI{:}) = repmat(...
                bsxfun(@times,...
                    shiftdim(kSubRangeOut(:), 1-outDimIdx - 2),... 
                    shiftdim(kSubRangeIn(:), 1-inDimIdx - 2)...
                ), [1 1 nbCopies] );
        end
    end % K x K

    % Divide by K^2
    K2 = getK2(dataSize, samplePitch);
    result = bsxfun(@rdivide, result, shiftdim(K2 + (K2==0), -2)); % avoid the NaN at the origin => replace with zero
    result(:,:,K2==0) = eye(nbDimsOut,nbDimsOut); % undefined origin => project the DC as longitudinal
end
function result = getPiTFt(dataSize, samplePitch)
    nbDimsOut = 3; % Maintain field in higher dimensions than the sample
    
    result = bsxfun(@minus, eye(nbDimsOut,nbDimsOut), getPiLFt(dataSize, samplePitch));
end
function result = longitudinalProjectorFtFunction(xFt, samplePitch)
    dataSize = size(xFt);
    nbInputDims = dataSize(1);
    dataSize = dataSize(2:end);
    nbDims = min([numel(dataSize), numel(samplePitch)]);
    nbOutputDims = max(nbDims, nbInputDims);

    kRanges = arrayfun(@(dimIdx) ifftshift(2*pi*calcRanges(dataSize(dimIdx), 1./(dataSize(dimIdx).*samplePitch(dimIdx)))), [1:nbDims], 'UniformOutput', false);

    % (K x K) . xFt == K x (K . xFt)
    projection = zeros([1, dataSize], class(xFt));
    K2 = projection;
    ftRangeI = arrayfun(@(len) (1:len), dataSize, 'UniformOutput',false);
    % Position-wise projection of xFt on k vectors
    for inDimIdx = 1:nbDims,
        kRange = shiftdim(kRanges{inDimIdx}(:), 1-inDimIdx - 1);
        projection = projection + bsxfun(@times, kRange, xFt(inDimIdx, ftRangeI{:}) );
        K2 = bsxfun(@plus, K2, kRange.^2);
    end
    % Divide by K^2
    projection = projection ./ (K2 + (K2==0)); % avoid the NaN at the origin => replace with zero
    zeroK2 = find(K2 == 0);
    clear K2;
    
    result = zeros([nbOutputDims, dataSize], class(xFt));
    for outDimIdx = 1:nbDims  % Saving memory by not storing the complete tensor
        % stretch each k vector to be as long as the projection
        result(outDimIdx, ftRangeI{:}) = bsxfun(@times, shiftdim(kRanges{outDimIdx}(:), 1-outDimIdx - 1), projection);
    end
    
    result(:, zeroK2) = xFt(:, zeroK2); % undefined origin => project the DC as longitudinal

end % PiLF <- (K x K)/K^2

% Helper function for calculation of the Fourier transform of the Green function
function K2 = getK2(dataSize, samplePitch)
    K2 = 0;
    for dimIdx = 1:numel(dataSize)
        kSubRange = 2*pi * calcFrequencyRanges(calcRanges(dataSize(dimIdx), samplePitch(dimIdx)));
        K2 = bsxfun(@plus, K2, shiftdim(kSubRange(:).^2, 1-dimIdx));
    end
end
function K = getK(dataSize, samplePitch)
    nbDims = numel(dataSize);
    K = zeros([nbDims, dataSize]);
    for dimIdx = 1:nbDims
        kSubRange = 2*pi * calcFrequencyRanges(calcRanges(dataSize(dimIdx), samplePitch(dimIdx)));
        repSize = dataSize;
        repSize(dimIdx) = 1;
        K(dimIdx,:) = reshape(repmat(shiftdim(kSubRange(:), -dimIdx), [1, repSize]), 1, []);
    end
end
% N-1 dimensional Fast Fourier Transform on all but the first dimension
function result = fftnButDim1Function(F)
    result = zeros(size(F), class(F));
    rangeI = arrayfun(@(len) (1:len), size(F), 'UniformOutput',false);
    rangeI = rangeI(2:end);
    for inDimIdx = 1:size(F,1)
        result(inDimIdx,rangeI{:}) = fftn(F(inDimIdx,rangeI{:})); % First dimension is singleton, ignored by fftn
    end
end
% N-1 dimensional Inverse Fast Fourier Transform on all but the first dimension
function result = ifftnButDim1Function(FFt)
    result = zeros(size(FFt), class(FFt));

    fRangeI = arrayfun(@(len) (1:len), size(FFt), 'UniformOutput',false);
    fRangeI = fRangeI(2:end);
    for outDimIdx = 1:size(FFt,1)
        result(outDimIdx,fRangeI{:}) = ifftn(FFt(outDimIdx,fRangeI{:})); % Move the Fourier transform outside of the 'dot' product
    end
end