% result = arrayMatMul(A, B)
%
% Parallel matrix product on the first two dimensions of A and B
%
function result = arrayMatMul(A, B)
  % Get data-sizes from both input arrays
  dataSizeA = size(A);
  ASize = dataSizeA(1:2);
  dataSizeA = dataSizeA(3:end);
  dataSizeB = size(B);
  BSize = dataSizeB(1:2);
  dataSizeB = dataSizeB(3:end);

  isScalarArrayA = all(ASize == 1);
  isScalarArrayB = all(BSize == 1);
  if ~isScalarArrayA && ~isScalarArrayB
    % one-pad up to the same length
    dataSizeA(end+1:max(numel(dataSizeA), numel(dataSizeB))) = 1;
    dataSizeB(end+1:max(numel(dataSizeA), numel(dataSizeB))) = 1;
    % determine singleton expansion size
    dataSize = (dataSizeA > 1) .* dataSizeA + (dataSizeA <= 1) .* dataSizeB;
    % Create space for result
    result = zeros([ASize(1) BSize(2) dataSize], class(A));
    for rowIdx = 1:ASize(1)
      sliceA = permute(A(rowIdx,:,:), [2 1 3]);
      for colIdx = 1:BSize(2)
        result(rowIdx,colIdx,:) = sum(bsxfun(@times, sliceA, B(:,colIdx,:)), 1);
      end
    end
  else % treat scalar as scaled identity matrix
    result = bsxfun(@times, A, B);
  end
end