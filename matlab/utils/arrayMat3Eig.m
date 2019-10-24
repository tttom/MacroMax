% D = arrayMat3Eig(A)
%
% Calculates the eigenvalues of the 3x3 matrices represented by A and
% returns a new array of 3-vectors, one for each matrix in A and of the
% same dimensions, baring the second dimension. When the first two
% dimensions are 3x1 or 1x3, a diagonal matrix is assumed. When the first
% two dimensions are singletons (1x1), a constant diagonal matrix is assumed and only one eigenvalue is returned.
% All dimensions are maintained in size except for dimension 2
%
function D = arrayMat3Eig(A)
    if nargin < 1
        A = randn(3,3,5,6,7);
        A = A + 1i*randn(size(A));
    end
    
    dataSize = size(A);
    dataSize = dataSize(3:end);
    if size(A,1)==3 && size(A,2)==3
        % A 3x3 matrix in the first two dimensions
        C = zeros([4, 1, dataSize], class(A));
        C(1,:) = A(1,1,:).*(A(2,2,:).*A(3,3,:) - A(2,3,:).*A(3,2,:)) ...
                - A(1,2,:).*(A(2,1,:).*A(3,3,:) - A(2,3,:).*A(3,1,:)) ...
                + A(1,3,:).*(A(2,1,:).*A(3,2,:) - A(2,2,:).*A(3,1,:));
        C(2,:) = - A(1,1,:).*(A(2,2,:) + A(3,3,:)) + A(1,2,:).*A(2,1,:) + A(1,3,:).*A(3,1,:) - A(2,2,:).*A(3,3,:) + A(2,3,:).*A(3,2,:);
        C(3,:) = A(1,1,:) + A(2,2,:) + A(3,3,:);
        C(4,:) = -1;
        
        C = reshape(C, [4 dataSize 1]);
        D = calcRootsOfLowOrderPolynomial(C);
    else
        sizes = [size(A,1) size(A,2)];
        if all(sizes == 1 | sizes == 3)
            % Maybe a scalar per or diagonal-as-column
            D = reshape(A, [prod(sizes), 1, dataSize]);
        else
            error('The first two dimensions of the input array should be either of length 1 or of length 3.');
        end
    end
    
%     % Debug output
%     if nargout < 1
%         [~, Dref] = eig(A(:,:,1+floor(end/2)));
%         Dref = sort(diag(Dref));       
%         calculationError = (abs(sort(D(:,1+floor(end/2))) - Dref)./abs(Dref)).'
%         
%         clear D;
%     end
end