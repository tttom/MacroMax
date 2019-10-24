% X = calcRootsOfLowOrderPolynomial(C)
%
% Calculates the (complex) roots of polynomials up to order 3 in parallel.
% The coefficients of the polynomials are represented in A as column vectors,
% one for each polynomial to determine the roots of. The coefficients are
% input for low to high order in each column. In other words, the polynomial is:
%    sum(C.*x^[0:numel(C)-1]) == 0
%
%
function X = calcRootsOfLowOrderPolynomial(C)
    if nargin<1,
%         C = [24+3i -20+34i 34i 23; 1 2 3 4].';
        C = randn([4 3 2]) + 1i*randn([4 3 2]);
    end
    
    inputSize = size(C);
    outputSize = inputSize;
    outputSize(1) = outputSize(1) - 1;
    
    switch size(C, 1)
        case {0,1}
            X = [];
        case 2
            %C(1) + C(2)*X == 0
            X = -C(1,:) ./ C(2,:);
        case 3
            %C(1) + C(2)*X + C(3)*X.^2 == 0
            d = C(2,:).^2 - 4*C(3,:).*C(1,:);
            X = bsxfun(@rdivide, bsxfun(@plus, -C(2,:), bsxfun(@times, [-1; 1], sqrt(d))), 2*C(3,:));
        case 4
            % Based on Ch 5.6 in NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING (ISBN 0-521-43108-5)
            % The equation first appears in Chapter VI of Francois Viete’s treatise "De emendatione," published in 1615
            %
            a = C(3,:)./C(4,:)./3;
            b = C(2,:)./C(4,:);
            c = C(1,:)./C(4,:);
            
            Q = a.^2 - b./3;
            R = a.^3 -a.*b./2 + c./2;
            clear b c;

            % General procedure:
            S2 = R.^2 - Q.^3;
            realRoots = real(S2) < eps(class(S2));
            if ~all(realRoots(:))
              S = sqrt(S2);
              clear S2;
              signNonZero = @(x) 2*(x>=0) - 1;
              A = -(R + signNonZero(real(conj(R).*S)).*S).^(1/3);
              % choose sign of sqrt so that real(conj(R) .* sqrt(R.^2 - Q.^3)) >= 0
              B = (A~=0) .* Q./(A + (A==0));
              X = bsxfun(@minus,...
                  bsxfun(@times, A, exp(2i*pi * [-1:1].'./3))...
                  + bsxfun(@times, B, exp(-2i*pi * [-1:1].'./3)),...
                      a);
            else
              % All roots are real, the following should be more accurate:
              clear S2;
              R = real(R);
              smallQ = abs(Q) < eps(class(Q));
              theta = acos(R .* (Q + smallQ).^(-3/2)); % If Q is zero, it doesn't matter what theta is
              X = bsxfun(@minus, -2*bsxfun(@times, sqrt(Q), cos(bsxfun(@plus, theta, 2*pi*[-1:1].')/3)), a);
            end
          %
          % See also: https://arxiv.org/pdf/physics/0610206.pdf
          % % Eigenvalues lambda of 3x3 Hermitian matrix A
          %   a = - A(1,1,:) - A(2,2,:) - A(3,3,:);
          %   b = A(1,1,:).*A(2,2,:) + A(1,1,:).*A(3,3,:) + A(2,2,:).*A(3,3,:) - abs(A(1,2,:)).^2 - abs(A(1,3,:)).^2 - abs(A(2,3,:)).^2;
          %   c = A(1,1,:).*abs(A(2,3,:)).^2 + A(2,2,:).*abs(A(1,3,:)).^2 + A(3,3,:).*abs(A(1,2,:)).^2 - A(1,1,:).*A(2,2,:).*A(3,3,:) - 2*real(conj(A(1,3,:)).*A(1,2,:).*A(2,3,:));
          % 
          %   p = a.^2 - 3*b;
          %   q = -27/2*c - a^3 + 9/2*a*b;
          %   phi = atan2(sqrt(27*(b.^2.*(p-b)./4 + c.*(q + 27/4*c))), q);
          % 
          %   X = 2 * cos((phi + [-1:1]*2*pi)/3);
          %   lambda = (sqrt(p) * X - a)./3;
          %
          
        otherwise
            error('Orders above 3 not implemented!');
    end
    
    X = reshape(X, outputSize);
    
%     result = evaluatePolynomial(C, X);
%     rootSize = max(abs(result(:)))
end
%
% Evaluates the polynomial for testing
%
function results = evaluatePolynomial(C, X)
    outputSize = size(X);
    
    results = 0;
    for idx = size(C,1):-1:1
        results = bsxfun(@plus, results.*X, reshape(C(idx,:), [1 outputSize(2:end)]));
    end
end