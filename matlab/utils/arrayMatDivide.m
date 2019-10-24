% result = arrayMatDivide(A, B)
%
% Parallel matrix left division, A^{-1}B, on the first two dimensions of A and B
% result = A \ B
% Singleton dimensions are replicated unless the first two dimensions are
% zero, in which case a scaled identity matrix is assumed.
%
% A must be square in the first two dimensions, the result is of size:
% [size(A,1), size(B,2), ...]
%
function result = arrayMatDivide(A, B)
  if nargin < 1
%     A = [0 3i 0; -2 0 0; 0 0 1];
    A = repmat([0 1 0; -1 0 0; 0 0 1], [1 1 1 2]);
  end
  if nargin < 2
    B = repmat(eye(3), [1 1 2 1]);
  end
  dataSizeA = size(A);
  ASize = dataSizeA(1:2);
  dataSizeA = dataSizeA(3:end);
  dataSizeB = size(B);
  BSize = dataSizeB(1:2);
  dataSizeB = dataSizeB(3:end);
  dataSizeA(end+1:numel(dataSizeB)) = 1;
  dataSizeB(end+1:numel(dataSizeA)) = 1;
  dataSize = [max(dataSizeA, dataSizeB), 1];
  isScalarArrayA = all(ASize == 1);
  if ~isScalarArrayA
    isScalarArrayB = all(BSize == 1);
    if isScalarArrayB
      % If scalar instead of matrix, assume it is the scalar*idenity matrix
      for dimIdx = ASize(1):-1:1
        B(dimIdx, dimIdx, :) = B(1,1,:);
      end
      BSize = [1 1]*ASize(1);
    end
    if any(dataSizeA > dataSizeB)
      B = repmat(B, [1 1 max(1, dataSizeA./dataSizeB)]); % Replicate if singleton dimension
    end
    if any(dataSizeB > dataSizeA)
      A = repmat(A, [1 1 max(1, dataSizeB./dataSizeA)]); % Replicate if singleton dimension
    end
    % Do Gaussian elimination
    result = cat(2, A(:,:,:), B(:,:,:)); % flatten in dim 3
    ABSize = ASize + BSize.*[0, 1];
    for colIdx = 1:ASize(2)
      rowIdx = colIdx;
      remainingColumns = [colIdx+1:ABSize(2)];
      % Do partial pivoting
      [~, maxSub] = max(abs(result([rowIdx:ABSize(1)], colIdx, :)), [], 1);
      maxSub = maxSub + rowIdx-1;
      maxI = bsxfun(@plus, maxSub, ABSize(1)*([1:ABSize(2)]-1) ); % Index whole row
      maxI = bsxfun(@plus, maxI, prod(ABSize)*shiftdim(([1:prod(dataSize)]-1),-1) ); % Index whole data set
      maxElements = result(maxI);
      % swap rows
      result(maxI) = result(rowIdx, :, :);
      result(rowIdx, :, :) = maxElements;

      allOtherRows = [1:rowIdx-1, rowIdx+1:ABSize(1)];
      % Scale the pivot row so that all (rowIdx, colIdx,:) == 1,
      % ignore columns before colIdx
      result(rowIdx, remainingColumns, :) = bsxfun(@rdivide, result(rowIdx, remainingColumns, :), result(rowIdx, colIdx, :) );
      % Subtract a scaled version of row rowIdx of all other rows, ignore column colIdx and before 
      result(allOtherRows, remainingColumns, :) = result(allOtherRows, remainingColumns, :)...
        - bsxfun(@times, result(allOtherRows, colIdx, :), result(rowIdx, remainingColumns, :) );
    end
    result = reshape(result(:, (ASize(2)+1):ABSize(2), :), [ABSize(1), ABSize(2)-ASize(2), dataSize]); % unflatten
  else % treat scalar as scaled identity matrix
      result = bsxfun(@rdivide, B, A);
  end
end