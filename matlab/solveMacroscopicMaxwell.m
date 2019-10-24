%
% [E, rmsError, itIdx] = solveMacroscopicMaxwell(ranges, k0, epsilon, xi, zeta, mu, sourceDistribution, progressFunction, E, lowPassBands)
%
% Calculates the vector field, E, that satisfies the Helmholtz equation for a
% potential and source's current density distribution given by chi and calcS at the coordinates specified by the ranges.
%
% The direct implementation keeps everything in memory all the time.
% This function assumes periodic boundary conditions. Add absorbing or PML
% layers to simulate other situations.
%
% Input parameters:
%     ranges: a cell array of ranges to calculate the solution at. In the
%             case of 1D, a simple vector may be provided. The length of the ranges
%             determines the data_shape, which must match the dimensions of
%             (the output of) epsilon, xi, zeta, mu, source, and the
%             optional start field, unless these are singletons.
%
%     k0: the wavenumber in vacuum = 2pi / wavelength.
%             The wavelength in the same units as used for the other inputs/outputs.
%
%     epsilon: an array or function that returns the (tensor) permittivity at the
%             points indicated by the ranges specified as its input arguments.
%     xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the
%             points indicated by the ranges specified as its input arguments.
%     zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the
%             points indicated by the ranges specified as its input arguments.
%     mu: an array or function that returns the (tensor) permeability at the
%             points indicated by the ranges specified as its input arguments.
%
%     sourceDistribution: an array or function that returns the (vector) source current density at the
%             points indicated by the ranges specified as its input arguments. The
%             source values relate to the current density, j, as source = 1i *
%             angularFrequency * Const.mu_0 * j
%
%     progressFunction: if specified (and not []), a function called after
%             every iteration. The function should return true if the iteration is to continue, false otherwise.
%             If instead of a function handle numeric values are provided, these are interpreted as the stop criterion
%             maximum iteration and root-mean-square error as a 2-vector.
%
%     E: (optional) the (vector) field to start the calculation from.
%             Anything close to the solution will reduce the number of iterations
%             required.
%
%     lowPassBands: (optional, default: 'none') flag to indicate how to band
%             limit the inputs and calculation to prevent aliasing. It can
%             be specified as a vector with the low-pass fractions per band:
%             [final, source, material, iteration], where final is the
%             fraction of the final output to be retained, source that of
%             the source at the start of the calculation, material that of
%             the epsilon, xi, zeta, and mu, and iteration indicates
%             whether to band limit at each step of the process. Note that
%             the band-limiting of the material properties may introduce
%             gain and thus divergence. The strings 'none', 'source',
%             'susceptibility', 'input', 'iteration', can be provided as
%             shorthands for 50% filtering of the corresponding.
%
% Output parameters:
%     resultE: The solution field as an n-d array. If the source is
%             vectorial, the first dimension is 3. If the source is scalar,
%             the first dimension corresponds to the first range in ranges.
%     rmsError: an estimate of the relative error
%     itIdx: the iteration number at which the algorithm terminated
%
function [resultE, rmsError, itIdx] = solveMacroscopicMaxwell(ranges, k0, epsilon, xi, zeta, mu, sourceDistribution, progressFunction, E, lowPassBands)
    if nargin < 1
      ranges = cell(1,0);
    end
    if ~iscell(ranges)
      ranges = {ranges};
    end
    numericType = class(ranges{1});
    data_shape = cellfun(@(r) ones(1, numericType)*numel(r), ranges, 'UniformOutput', true);
    samplePitch = cellfun(@(r) ones(1, numericType)*diff(r(1:min(2,end))), ranges, 'UniformOutput', true);
    if nargin < 2
      k0 = 1;
    end
    if 2*pi / k0 < 2*min(samplePitch)
      message = logMessage('The wavenumber k0=%d should be no larger than %d for sample pitch of %d.', [k0 2*pi/(2*min(samplePitch)) min(samplePitch)]);
      warning(message);
    end
    if nargin < 3 || isempty(epsilon)
      epsilon = @vacuumPermittivity;
    end
    if nargin < 4 || isempty(xi)
      xi = 0;
    end
    if nargin < 5 || isempty(zeta)
      zeta = 0;
    end
    if nargin < 6 || isempty(mu)
      mu = 1;
    end
    if nargin < 7 || isempty(sourceDistribution)
      sourceDistribution = @originSource;
    end
    if nargin < 8
      progressFunction = [1000, 1e-3]; % stop criterion [maxIt, rms]
    end
    if isnumeric(progressFunction)
      if numel(progressFunction) > 1
        progressFunction = @(itIdx, rmsError) itIdx < progressFunction(1) && rmsError > progressFunction(2); % Just check if accuracy or the maximum iteration count reached
      else
        progressFunction = @(itIdx) itIdx > progressFunction(1); % Just check if the maximum iteration count reached
      end
    end
    if nargin < 9
      E = [];
    end
    if nargin < 10 || isempty(lowPassBands)
      lowPassBands = 'none';
    end
    
    if ischar(lowPassBands)
      if strcmpi(lowPassBands(1:min(2,end)), 'no')
        lowPassBands = [1 1 1 1];
      elseif strcmpi(lowPassBands, 'final')
        lowPassBands = [0.5 1 1 1];
      elseif strcmpi(lowPassBands(1:min(6,end)), 'source')
        lowPassBands = [1 0.5 1 1];
      elseif strcmpi(lowPassBands(1:min(3,end)), 'mat') || strcmpi(lowPassBands(1:min(3,end)), 'sus')
        lowPassBands = [1 1 0.5 1];
      elseif strcmpi(lowPassBands(1:min(5,end)), 'input')
        lowPassBands = [1 0.5 0.5 1];
      elseif strcmpi(lowPassBands(1:min(4,end)), 'iter')
        lowPassBands = [1 1 1 0.5];
      elseif strcmpi(lowPassBands, 'all')
        lowPassBands = [0.5 0.5 0.5 0.5];
      else
        logMessage('Unrecognized low pass definition: %s', lowPassBands);
        lowPassBands = [1 1 1 1];
      end
    end
    if any(lowPassBands <= 0)
      logMessage('Bandwidth set to zero, assuming infinity is meant.');
      lowPassBands(lowPassBands <= 0) = Inf;
    end
    lowPassFinal = lowPassBands(1);
    lowPassSource = lowPassBands(2);
    lowPassSusceptibility = lowPassBands(3);
    lowPassIteration = lowPassBands(4);
    
    if isa(epsilon, 'function_handle')
      epsilon = epsilon(ranges{:});
    end
    if isa(xi, 'function_handle')
      xi = xi(ranges{:});
    end
    if isa(zeta, 'function_handle')
      zeta = zeta(ranges{:});
    end
    if isa(mu, 'function_handle')
      mu = mu(ranges{:});
    end
    epsilon = fixShape(epsilon, data_shape);
    xi = fixShape(xi, data_shape);
    zeta = fixShape(zeta, data_shape);
    mu = fixShape(mu, data_shape);
    
    maxKernelSizeInPx = data_shape ./ 4;
    maxKernelResidue = 1 / 100;
    
    % Limit the spatial frequencies to avoid aliasing.
    projF = getVectorialProjectors(samplePitch .* k0);
    function fld = lowPass(fld, fraction)
      if fraction < 1.0
        fldFt = projF.fftnButDim1(fld);
        for dimIdx = 1:numel(ranges)
          fRange = calcFrequencyRanges(ranges{dimIdx});
%           filt = abs(fRange) < -fRange(1+floor(end/2)) * fraction; % sharp cut-off
%           filt = max(0, min(1, 2 - abs(fRange) ./ (-fRange(1+floor(end/2)) * fraction))); % linear roll-off
          normalizedFreq = abs(fRange) ./ (-fRange(1+floor(end/2)));
          filt = (normalizedFreq < fraction) +...
            (normalizedFreq >= fraction & normalizedFreq < 1.5*fraction)...
            .* (1 + cos(pi * (normalizedFreq-fraction) ./ min(1-fraction, (1.5-1)*fraction) )) ./ 2; % Cosine roll-off over half the remainder
          fldFt = bsxfun(@times, fldFt, shiftdim(filt(:), -1 - dimIdx));
        end
        fld = projF.ifftnButDim1(fldFt);
      end
    end
    % Make sure that the material properties won't cause aliasing 
    if ~isscalar(epsilon)
      epsilon = lowPass(epsilon, lowPassSusceptibility);
    end
    if ~isscalar(xi)
      xi = lowPass(xi, lowPassSusceptibility);
    end
    if ~isscalar(zeta)
      zeta = lowPass(zeta, lowPassSusceptibility);
    end
    if ~isscalar(mu)
      mu = lowPass(mu, lowPassSusceptibility);
    end
      
    maxKernelRadiusInRad = min(maxKernelSizeInPx./samplePitch)/2 / k0;
    [alpha, beta] = calcBiases(epsilon, xi, zeta, mu, samplePitch, k0, maxKernelRadiusInRad, maxKernelResidue);
    [chiMul, gammaMul, solutionPropagationStepEstimate] = calcWorkingSusceptibility(alpha, beta, epsilon, xi, zeta, mu, samplePitch, k0);
    
    if isa(sourceDistribution, 'function_handle')
        sourceDistribution = sourceDistribution(ranges{:});
    end
    sourceDistribution = fixShape(sourceDistribution, data_shape);
    sourcePolVectorSize = size(sourceDistribution,1);
    if sourcePolVectorSize > 3
        warning('Source polarization has dimension of %d', sourcePolVectorSize);
    end
    scatterVectorSize = size(chiMul(sourceDistribution(:,1)), 1);
    polVectorSize = max(sourcePolVectorSize, scatterVectorSize); % Should be the same in principle
    if polVectorSize > 1
        % If the source polarization has dimension 1, it is assumed to be scalar waves
        % Even if the source is low dimensional polarized, it can scatter in 3D.
        polVectorSize = max(polVectorSize, numel(ranges));
    end
    isVectorialCalculation = polVectorSize > 1;
    
    if isVectorialCalculation
        output_shape = [polVectorSize data_shape];
    else
        output_shape = data_shape;
    end

    function convolveDyadicG = calcG(alpha)
      % Calculate the convolution filter once
      gScalarFt = 1./(shiftdim(projF.K2Functor(data_shape), -2) - alpha);
      function FFt = gFtMul(FFt) % No need to represent the full matrix in memory
          if isVectorialCalculation
              PiLFFt = projF.longitudinalProjectorFt(reshape(FFt,[size(FFt,1) [data_shape 1]])); % Collapses second dimension and creates K.^2 on-the-fly and still memory intensive
              PiLFFt = reshape(PiLFFt, [size(PiLFFt,1) 1 data_shape]);
              FFt = bsxfun(@times, gScalarFt, FFt - PiLFFt) - PiLFFt./alpha;
          else
              FFt = bsxfun(@times, gScalarFt, FFt);
          end
      end
      function F = GMul(F)
          % Convert each component separately to frequency coordinates
          FFt = projF.fftnButDim1(F);

          FFt = gFtMul(FFt); % Memory heavy

          % Back to spatial coordinates        
          F = projF.ifftnButDim1(FFt);
      end
      convolveDyadicG = @GMul;
    end
    convolveDyadicG = calcG(alpha);
    
    volumeSize = data_shape .* samplePitch;
    logMessage('Solution propagation speed: %0.0fnm/iteration in volume of width %0.3fum.', [solutionPropagationStepEstimate*1e9 max(volumeSize)*1e6]);
    nbIterationsExpected = ceil(2*norm(volumeSize./solutionPropagationStepEstimate)); % the longest axis
    logMessage('Expected number of iterations required: %d', nbIterationsExpected);
    
    % Initial state
    if isa(E, 'function_handle')
        E = E(ranges{:});
    end
    if numel(E) > 1
        E = fixShape(E, data_shape);
    end
    
    % Convert the source distribition to the working source distribution
    S = sourceDistribution ./ beta; % Adjust for magnetic bias, if any
    clear sourceDistribution;
        
    itIdx = 1;
    if ~all(S(:) == 0)
      S = lowPass(S, lowPassSource);
      
      % Determine the residue normalization
      precS = gammaMul(convolveDyadicG(S));
      normPrecS = norm(precS(:));  % The norm of the preconditioned source
      % If no start field provided, do a simplified first step of the iteration
      if isempty(E) || all(E(:) == 0)
          Ek02 = precS;
      else
          Ek02 = (k0^2) * E; % temporarily multiply by k0^2
      end
      clear E;
      
      Ek02 = lowPass(Ek02, lowPassIteration);
      
            
      %
      % Iteration loop
      %
      cont = true;
      previousUpdateNorm = Inf;
      narginUpdateFunctor = nargin(progressFunction);
      while cont
          % The actual iteration calculation: correction = i/alpha_i chi (G x (chi E + S) - E)
          correction = chiMul(Ek02);  % up to 4 FT
          correction = correction + S;
          correction = convolveDyadicG(correction);  % 2 FT
          correction = correction - Ek02;
          correction = gammaMul(correction);  % up to 4 FT
          % % The above as a one-liner:
          % correction = gammaMul(convolveDyadicG(bsxfun(@plus, chiMul(Ek02), S)) - Ek02);
  
          correction = lowPass(correction, lowPassIteration);
          
          % Check convergence criterion
          currentUpdateNorm = norm(correction(:));
          normIncrease = currentUpdateNorm / previousUpdateNorm;
          if normIncrease < 1
            previousUpdateNorm = currentUpdateNorm;
          
            % No divergence detected, updating field now.
            Ek02 = Ek02 + correction;
          else
            logMessage('Convergence criterion not fullfilled!\nEigenvalue of M is at least as large as %d!', normIncrease);
            
            % Increase alpha_i
            alpha = alpha + 0.50i * imag(alpha);
            logMessage('Increasing the imaginary part of the background permittivity to %d+%di...', [real(alpha), imag(alpha)]);
            % Determine new operators
            [chiMul, gammaMul] = calcWorkingSusceptibility(alpha, beta, epsilon, xi, zeta, mu, samplePitch, k0);
            % Update G
            convolveDyadicG = calcG(alpha);
          end
          
          % Handle per-iteration output
          argIn = {itIdx};
          if narginUpdateFunctor >= 2
              % Estimate the relative error
              rmsError = norm(correction(:)) / normPrecS;
              argIn{2} = rmsError;
          end
          if narginUpdateFunctor >= 3
              argIn{3} = (k0^-2) * reshape(Ek02, [output_shape 1]);
          end
          % Execute per-iteration caller logic
          cont = progressFunction(argIn{:});

          itIdx = itIdx + 1;
      end  % end of while loop
      
      % Low-pass the result if requested and not yet done during the iteration
      if lowPassFinal < lowPassIteration
        Ek02 = lowPass(Ek02, lowPassFinal);
      end
      
      resultE = (k0^-2) * reshape(Ek02, [output_shape 1]);
    else
      logMessage('The source current distribution is zero everywhere. The result field will be zero everywhere too.');
      resultE = zeros([output_shape 1], class(S));
      rmsError = 0;
    end
end

% Returns permittivity bias or susceptibility scale for fastest convergence
function [alpha, beta] = calcBiases(epsilon, xi, zeta, mu, samplePitch, k0, maxKernelRadiusInRad, maxKernelResidue)
  logMessage('Determining optimal permittivity bounds...');
  
  isMagnetic = any(xi(:) ~= 0) || any(zeta(:) ~= 0) || any(abs(mu(:) - mu(1)).^2 > 0);
  if ~isMagnetic && ~isscalar(mu)
    mu = mu(:,1);
  end
  isFullTensor = @(A) size(A,1)>=3;
    
  globalMax = @(C) abs(max(C(:)));
  transpose = @(T) conj(permute(T, [2 1 3:ndims(T)]));
  largestEigenvalue = @(M) globalMax(arrayMat3Eig(M)); 
  largestSingularvalue = @(M) sqrt(largestEigenvalue(arrayMatMul(transpose(M), M))); 
  
  % Do a quick check to see if the the media has no gain
  hasGain = @(A) -globalMax(-real(arrayMat3Eig(-1i/2 * (A - transpose(A))))) < -sqrt(eps(class(A)));
  if hasGain(epsilon)
    logMessage('Permittivity has gain. Cannot guarantee convergence!');
  end
  
  % Determine muInv
  if isFullTensor(mu)
    muInv = arrayMatDivide(mu, eye(3));
  else
    muInv = 1 ./ mu;
  end
  % Determine calcSigmaHH
  if isMagnetic
    muInvEye = eye(1 + 2*isFullTensor(muInv));
    muInvTranspose = transpose(muInv);
    muInv2 = arrayMatMul(muInvTranspose, muInv); % Positive definite 
    calcChiHHTheta2 = @(beta) bsxfun(@plus, muInv2, bsxfun(@minus, muInvEye .* abs(beta)^2, muInvTranspose .* beta + muInv .* conj(beta)) );
    calcSigmaHH = @(beta) sqrt(largestEigenvalue(calcChiHHTheta2(beta))) / abs(beta);
    
    if hasGain(mu) || hasGain(xi) || hasGain(zeta)
      logMessage('Permeability or bi-(an)isotropy has gain. Cannot guarantee convergence!');
    end
  else
    % non-magnetic, mu is scalar and both xi and zeta are zero
    beta = muInv;
    calcChiHH = @(beta) 1.0 - muInv / beta; % always zero when beta == muInv
    calcSigmaHH = @(beta) abs(calcChiHH(beta)); % always zero when beta == muInv
  end
    
  % Determine: calcChiEE, chiEHTheta, chiHETheta, chiHH, alpha, beta
  if xi ~= 0
    chiEHTheta = -1i * arrayMatMul(xi, muInv);
  else
    chiEHTheta = 0;
  end
  if zeta ~= 0
    chiHETheta = 1i * arrayMatMul(muInv, zeta);
  else
    chiHETheta = 0;
  end
  clear muInv;
  xiMuInvZeta = arrayMatMul(xi, -1i * chiHETheta);
  if isFullTensor(xiMuInvZeta) && ~isFullTensor(epsilon)
    epsilon = bsxfun(@times, epsilon, eye(3));
  end
  if isFullTensor(epsilon) && ~isFullTensor(xiMuInvZeta)
    xiMuInvZeta = bsxfun(@times, xiMuInvZeta, eye(3));
  end
  epsilonXiMuInvZeta = bsxfun(@minus, epsilon, xiMuInvZeta);
  clear xiMuInvZeta;
  epsilonXiMuInvZetaEye = eye(1 + 2*isFullTensor(epsilonXiMuInvZeta));
  epsilonXiMuInvZetaTranspose = transpose(epsilonXiMuInvZeta);
  epsilonXiMuInvZeta2 = arrayMatMul(epsilonXiMuInvZetaTranspose, epsilonXiMuInvZeta); % Positive definite
  calcDeltaEETheta2 = @(alpha, beta) bsxfun(@plus,...
      epsilonXiMuInvZeta2,...
      bsxfun(@minus, (abs(real(alpha)*beta)^2) * epsilonXiMuInvZetaEye,...
                      epsilonXiMuInvZetaTranspose * (real(alpha)*beta) + conj(real(alpha)*beta) * epsilonXiMuInvZeta)...
    );
  calcSigmaEE = @(alpha, beta) sqrt(largestEigenvalue(calcDeltaEETheta2(alpha, beta))) / abs(beta);
  % Determine alpha and beta
  if isMagnetic
    % Optimize the real part of alpha and beta
    sigmaD = norm(2*pi./(2*samplePitch)) / k0;
    sigmaHETheta = largestSingularvalue(chiHETheta);
    sigmaEHTheta = largestSingularvalue(chiEHTheta);
    maxSingularValueSum = @(alpha, beta) sigmaD^2 * calcSigmaHH(beta) + sigmaD * (sigmaEHTheta + sigmaHETheta) / abs(beta) + calcSigmaEE(alpha, beta);
    
    beta_from_vec = @(etaThetaVec) etaThetaVec(2)^2; % enforce positivity
    targetFunctionVec = @(vec) maxSingularValueSum(vec(1), beta_from_vec(vec)) * beta_from_vec(vec);
    alpha_beta_vec0 = [0 1];
    [alpha_beta_vec, minValue] = fminsearch(targetFunctionVec, alpha_beta_vec0, optimset('TolX', 1e-4, 'TolFun', 1e-4, 'MaxIter',100, 'MaxFunEvals', 2*100, 'Display', 'none'));
    beta = beta_from_vec(alpha_beta_vec);
    alpha = alpha_beta_vec(1) + 1i * maxSingularValueSum(alpha_beta_vec(1), beta);
    
    clear beta_from_vec alpha_beta_vec largestSingularvalueSum
  else
    % non-magnetic
    % beta = muInv; % set earlier in case non-magnetic
    targetFunctionVec = @(alpha) calcSigmaEE(alpha, beta); % beta is fixed to muInv so that chiHH == 0
    alpha0_r = 0;
    [alpha_r, alpha_i] = fminsearch(targetFunctionVec, alpha0_r, optimset('TolX', 1e-4, 'TolFun', 1e-4, 'MaxIter',100, 'MaxFunEvals', 100, 'Display', 'none'));
    alpha = alpha_r + 1i * alpha_i;
    
    clear alpha0_r alpha_r alpha_i;
  end
    
  % Limit the kernel size by increasing maxSusceptibilityOffset so that:
  % -log(maxKernelResidue)/(imag(sqrt(centralPermittivity + 1i*maxSusceptibilityOffset))) == maxKernelRadiusInRad
  %
  %     -log(maxKernelResidue) / maxKernelRadiusInRad == imag(sqrt(centralPermittivity + 1i*maxSusceptibilityOffset)) == B
  %     B^4 + centralPermittivity*B^2 - maxSusceptibilityOffset^2/4 == 0
  %     -centralPermittivity +- sqrt(centralPermittivity^2 + maxSusceptibilityOffset^2) == 2 * B^2
  %     -centralPermittivity +- sqrt(centralPermittivity^2 + maxSusceptibilityOffset^2) == 2*(-log(maxKernelResidue) / maxKernelRadiusInRad)^2
  % Increase offset if needed to restrict kernel size
  centralPermittivity = real(alpha);
  susceptibilityOffset = imag(alpha);
  minSusceptibilityOffset = sqrt(max(0.0, (2*(-log(maxKernelResidue) / maxKernelRadiusInRad)^2 + centralPermittivity)^2 - centralPermittivity^2));
  susceptibilityOffset = max(minSusceptibilityOffset, susceptibilityOffset);
  alpha = real(alpha) + 1i * susceptibilityOffset;
  
  logMessage('alpha = %d + %di, beta = %d', [real(alpha), imag(alpha), beta]);
end
% Determine the reference permittivity and return the corresponding operator to multiply by the working susceptibility
function [susceptibilityMul, gammaMul, solutionPropagationStepEstimate] = calcWorkingSusceptibility(alpha, beta, epsilon, xi, zeta, mu, samplePitch, k0)
  isFullTensor = @(A) size(A,1)>=3;
  isMagnetic = any(xi(:) ~= 0) || any(zeta(:) ~= 0) || any(abs(mu(:) - mu(1)).^2 > 0);
  % Determine muInv
  if isFullTensor(mu)
    muInv = arrayMatDivide(mu, eye(3));
  else
    muInv = 1 ./ mu;
  end
  % Determine: calcChiEE, chiEHTheta, chiHETheta, chiHH, alpha, beta
  if xi ~= 0
    chiEHTheta = -1i * arrayMatMul(xi, muInv);
  else
    chiEHTheta = 0;
  end
  if zeta ~= 0
    chiHETheta = 1i * arrayMatMul(muInv, zeta);
  else
    chiHETheta = 0;
  end
  % Determine calcChiHH, calcSigmaHH
  if isMagnetic
    muInvEye = eye(1 + 2*isFullTensor(muInv));
    calcChiHH = @(beta) bsxfun(@minus, muInvEye, muInv ./ beta);
  else
    % non-magnetic, mu is scalar and both xi and zeta are zero
    calcChiHH = @(beta) 1.0 - muInv / beta; % always zero when beta == muInv
  end
  clear muInv
  xiMuInvZeta = arrayMatMul(xi, -1i * chiHETheta);
  if isFullTensor(xiMuInvZeta) && ~isFullTensor(epsilon)
    epsilon = bsxfun(@times, epsilon, eye(3));
  end
  if isFullTensor(epsilon) && ~isFullTensor(xiMuInvZeta)
    xiMuInvZeta = bsxfun(@times, xiMuInvZeta, eye(3));
  end
  epsilonXiMuInvZeta = bsxfun(@minus, epsilon, xiMuInvZeta);
  clear xiMuInvZeta;
  epsilonXiMuInvZetaEye = eye(1 + 2*isFullTensor(epsilonXiMuInvZeta));
  calcChiEE = @(alpha, beta) bsxfun(@minus, epsilonXiMuInvZeta ./ beta, alpha * epsilonXiMuInvZetaEye);
  
  chiHH = calcChiHH(beta);
    
  % Once susceptibilityOffset is fixed, we can calculate also chiEE:
  chiEE = calcChiEE(alpha, beta);
  
  % Introduce constants into the chi-s so that chiMul takes less steps
  chiHE = chiHETheta / beta;
  clear chiHETheta;
  chiEH = chiEHTheta / beta;
  clear chiEHTheta;
  function chiMulE = chiMul(E)
      % Calculate the matrix product chi =  [I, -1i*mulD] * [chiEE chiHE; chiEH chiHH] * [I; 1i*mulD]
      D = @(E) reshape(calcCurl(permute(E, [1, 3:ndims(E), 2]), samplePitch * k0), size(E));
      
      ED = D(E);
      chiE = bsxfun(@plus, arrayMatMul(chiEE, E), arrayMatMul(chiHE, ED));
      chiH = bsxfun(@plus, arrayMatMul(chiEH, E), arrayMatMul(chiHH, ED));
      chiMulE = bsxfun(@plus, chiE, D(chiH));
  end
  function chiMulE = chiElectricMul(E)
      chiMulE = arrayMatMul(chiEE, E);
  end
  if isMagnetic
    susceptibilityMul = @(E) chiMul(E);
  else
    susceptibilityMul = @(E) chiElectricMul(E);
  end
  
  nEstimate = sqrt(abs(real(alpha)));
  solutionPropagationStepEstimate = 2 * nEstimate/imag(alpha)/k0;
  
  gammaMul = @(E) 1i/imag(alpha) .* susceptibilityMul(E);
end

% Make the first two 1x1 or 3x1, moving anything else into higher dimensions
function A = fixShape(A, data_shape)
  dataNDims = find([2 data_shape]>1, 1, 'last') - 1;
  sizeA = size(A);
  ANDims = find(sizeA>1, 1, 'last');
  if ~isempty(ANDims) && ANDims > 1
    firstDims = sizeA(1:ANDims-dataNDims);
    firstDims(end+1:2) = 1; % add singleton dimensions
    A = reshape(A, [firstDims, data_shape]);
  end
end

%
% Default input argument functions 
%
function source = originSource(varargin)
    if ~isempty(varargin)
        dataType = class(varargin{1});
    else
        dataType = 'double';
    end
    data_shape = cellfun(@(r) numel(r), varargin, 'UniformOutput', true);
    source = zeros([3 1 data_shape], dataType);
    originIndexes = cellfun(@(r) find(abs(r) == min(abs(r)), 1), varargin, 'UniformOutput', false);
    source(3, 1, originIndexes{:}) = 1; % Ez-polarization
end
function chi = vacuumPermittivity(varargin)
    if ~isempty(varargin)
        dataType = class(varargin{1});
    else
        dataType = 'double';
    end
    data_shape = cellfun(@(r) numel(r), varargin, 'UniformOutput', true);
    chi = repmat(eye([3 3], dataType), [1 1 data_shape]);
end