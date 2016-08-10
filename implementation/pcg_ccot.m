function [x,flag,relres,iter,resvec,p,rho] = pcg_ccot(A,b,tol,maxit,M1,M2,x0,p,rho,varargin)
%This is a modified version of Matlab's pcg function.
%
%PCG   Preconditioned Conjugate Gradients Method.
%   X = PCG(A,B) attempts to solve the system of linear equations A*X=B for
%   X. The N-by-N coefficient matrix A must be symmetric and positive
%   definite and the right hand side column vector B must have length N.
%
%   X = PCG(AFUN,B) accepts a function handle AFUN instead of the matrix A.
%   AFUN(X) accepts a vector input X and returns the matrix-vector product
%   A*X. In all of the following syntaxes, you can replace A by AFUN.
%
%   X = PCG(A,B,TOL) specifies the tolerance of the method. If TOL is []
%   then PCG uses the default, 1e-6.
%
%   X = PCG(A,B,TOL,MAXIT) specifies the maximum number of iterations. If
%   MAXIT is [] then PCG uses the default, min(N,20).
%
%   X = PCG(A,B,TOL,MAXIT,M) and X = PCG(A,B,TOL,MAXIT,M1,M2) use symmetric
%   positive definite preconditioner M or M=M1*M2 and effectively solve the
%   system inv(M)*A*X = inv(M)*B for X. If M is [] then a preconditioner
%   is not applied. M may be a function handle MFUN returning M\X.
%
%   X = PCG(A,B,TOL,MAXIT,M1,M2,X0) specifies the initial guess. If X0 is
%   [] then PCG uses the default, an all zero vector.
%
%   [X,FLAG] = PCG(A,B,...) also returns a convergence FLAG:
%    0 PCG converged to the desired tolerance TOL within MAXIT iterations
%    1 PCG iterated MAXIT times but did not converge.
%    2 preconditioner M was ill-conditioned.
%    3 PCG stagnated (two consecutive iterates were the same).
%    4 one of the scalar quantities calculated during PCG became too
%      small or too large to continue computing.
%
%   [X,FLAG,RELRES] = PCG(A,B,...) also returns the relative residual
%   NORM(B-A*X)/NORM(B). If FLAG is 0, then RELRES <= TOL.
%
%   [X,FLAG,RELRES,ITER] = PCG(A,B,...) also returns the iteration number
%   at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,FLAG,RELRES,ITER,RESVEC] = PCG(A,B,...) also returns a vector of the
%   estimated residual norms at each iteration including NORM(B-A*X0).
%
%   Example:
%      n1 = 21; A = gallery('moler',n1);  b1 = A*ones(n1,1);
%      tol = 1e-6;  maxit = 15;  M = diag([10:-1:1 1 1:10]);
%      [x1,flag1,rr1,iter1,rv1] = pcg(A,b1,tol,maxit,M);
%   Or use this parameterized matrix-vector product function:
%      afun = @(x,n)gallery('moler',n)*x;
%      n2 = 21; b2 = afun(ones(n2,1),n2);
%      [x2,flag2,rr2,iter2,rv2] = pcg(@(x)afun(x,n2),b2,tol,maxit,M);
%
%   Class support for inputs A,B,M1,M2,X0 and the output of AFUN:
%      float: double
%
%   See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, MINRES, QMR,
%   SYMMLQ, TFQMR, ICHOL, FUNCTION_HANDLE.

%   Copyright 1984-2013 The MathWorks, Inc.

if (nargin < 2)
    error(message('MATLAB:pcg:NotEnoughInputs'));
end

% Determine whether A is a matrix or a function.
[atype,afun,afcnstr] = iterchk(A);
if strcmp(atype,'matrix')
    error('A must be a function');
    % Check matrix and right hand side vector inputs have appropriate sizes
    [m,n] = size(A);
    if (m ~= n)
        error(message('MATLAB:pcg:NonSquareMatrix'));
    end
    if ~isequal(size(b),[m,1])
        error(message('MATLAB:pcg:RSHsizeMatchCoeffMatrix', m));
    end
else
    m = sum(cellfun(@numel, b));
    n = m;
%     if ~iscolumn(b)
%         error(message('MATLAB:pcg:RSHnotColumn'));
%     end
end

% Assign default values to unspecified parameters
if (nargin < 3) || isempty(tol)
    tol = 1e-6;
end
warned = 0;
if tol <= eps
    warning(message('MATLAB:pcg:tooSmallTolerance'));
    warned = 1;
    tol = eps;
elseif tol >= 1
    warning(message('MATLAB:pcg:tooBigTolerance'));
    warned = 1;
    tol = 1-eps;
end
if (nargin < 4) || isempty(maxit)
    maxit = min(n,20);
end

% Check for all zero right hand side vector => all zero solution
n2b = norm_cdcf(b);                     % Norm of rhs vector, b
if (n2b == 0)                      % if    rhs vector is all zeros
    x = b;                % then  solution is all zeros
    flag = 0;                      % a valid solution has been obtained
    relres = 0;                    % the relative residual is actually 0/0
    iter = 0;                      % no iterations need be performed
    resvec = 0;                    % resvec(1) = norm(b-A*x) = norm(0)
    if (nargout < 2)
        itermsg('pcg',tol,maxit,0,flag,iter,NaN);
    end
    return
end

if ((nargin >= 5) && ~isempty(M1))
    existM1 = 1;
    [m1type,m1fun,m1fcnstr] = iterchk(M1);
    if strcmp(m1type,'matrix')
        if ~isequal(size(M1),[m,m])
            error(message('MATLAB:pcg:WrongPrecondSize', m));
        end
    end
else
    existM1 = 0;
    m1type = 'matrix';
end

if ((nargin >= 6) && ~isempty(M2))
    existM2 = 1;
    [m2type,m2fun,m2fcnstr] = iterchk(M2);
    if strcmp(m2type,'matrix')
        if ~isequal(size(M2),[m,m])
            error(message('MATLAB:pcg:WrongPrecondSize', m));
        end
    end
else
    existM2 = 0;
    m2type = 'matrix';
end

if ((nargin >= 7) && ~isempty(x0))
%     if ~isequal(size(x0),[n,1])
%         error(message('MATLAB:pcg:WrongInitGuessSize', n));
%     else
        x = x0;
%     end
    else
    error('Initial solution must be provided');
%     x = zeros(n,1);
end

% if ((nargin > 7) && strcmp(atype,'matrix') && ...
%         strcmp(m1type,'matrix') && strcmp(m2type,'matrix'))
%     error(message('MATLAB:pcg:TooManyInputs'));
% end

% Set up for the method
flag = 1;
% xmin = x;                          % Iterate which has minimal residual so far
% imin = 0;                          % Iteration at which xmin was computed
tolb = tol * n2b;                  % Relative tolerance
r = cellfun(@minus, b, iterapp('mtimes',afun,atype,afcnstr,x,varargin{:}), 'uniformoutput', false);
normr = norm_cdcf(r);                   % Norm of residual
normr_act = normr;

if (normr <= tolb)                 % Initial guess is a good enough solution
    flag = 0;
    relres = normr / n2b;
    iter = 0;
    resvec = normr;
    if (nargout < 2)
        itermsg('pcg',tol,maxit,0,flag,iter,relres);
    end
    return
end

resvec = zeros(maxit+1,1);         % Preallocate vector for norm of residuals
resvec(1,:) = normr;               % resvec(1) = norm(b-A*x0)
% normrmin = normr;                  % Norm of minimum residual

if nargin < 9 || isempty(rho)
rho = 1;
end

% stag = 0;                          % stagnation of the method
% moresteps = 0;
% maxmsteps = min([floor(n/50),5,n-maxit]);
% maxstagsteps = 3;

% loop over maxit iterations (unless convergence or failure)

for ii = 1 : maxit
    if existM1
        y = iterapp('mldivide',m1fun,m1type,m1fcnstr,r,varargin{:});
%         if ~all(isfinite(y))
%             flag = 2;
%             break
%         end
    else % no preconditioner
        y = r;
    end
    
    if existM2
        z = iterapp('mldivide',m2fun,m2type,m2fcnstr,y,varargin{:});
%         if ~all(isfinite(z))
%             flag = 2;
%             break
%         end
    else % no preconditioner
        z = y;
    end
    
    rho1 = rho;
    rho = inner_product_cdcf(r, z);
    if ((rho == 0) || isinf(rho))
        flag = 4;
        break
    end
    if (ii == 1 && (nargin < 8 || isempty(p)))
        p = z;
    else
        beta = rho / rho1;
        if ((beta == 0) || isinf(beta))
            flag = 4;
            break
        end
        p = cellfun(@(z,p) z + beta * p, z, p, 'uniformoutput', false);
    end
    q = iterapp('mtimes',afun,atype,afcnstr,p,varargin{:});
    pq = inner_product_cdcf(p, q);
    if ((pq <= 0) || isinf(pq))
        flag = 4;
        break
    else
        alpha = rho / pq;
    end
    if isinf(alpha)
        flag = 4;
        break
    end
    
    % Check for stagnation of the method    
%     if (norm_cdcf(p)*abs(alpha) < eps*norm_cdcf(x))
%         stag = stag + 1;
%     else
%         stag = 0;
%     end
    
    x = cellfun(@(x,p) x + alpha * p, x, p, 'uniformoutput', false);             % form new iterate
    r = cellfun(@(r,q) r - alpha * q, r, q, 'uniformoutput', false);
    normr = norm_cdcf(r);
    normr_act = normr;
    resvec(ii+1,1) = normr;
    
    % check for convergence
%     if (normr <= tolb || stag >= maxstagsteps || moresteps)
%         r = cellfun(@minus, b, iterapp('mtimes',afun,atype,afcnstr,x,varargin{:}), 'uniformoutput', false);
%         normr_act = norm_cdcf(r);
%         resvec(ii+1,1) = normr_act;
%         if (normr_act <= tolb)
%             flag = 0;
%             iter = ii;
%             break
%         else
%             if stag >= maxstagsteps && moresteps == 0
%                 stag = 0;
%             end
%             moresteps = moresteps + 1;
%             if moresteps >= maxmsteps
%                 if ~warned
%                     warning(message('MATLAB:pcg:tooSmallTolerance'));
%                 end
%                 flag = 3;
%                 iter = ii;
%                 break;
%             end
%         end
%     end
%     if (normr_act < normrmin)      % update minimal norm quantities
%         normrmin = normr_act;
%         xmin = x;
%         imin = ii;
%     end
%     if stag >= maxstagsteps
%         flag = 3;
%         break;
%     end
end                                % for ii = 1 : maxit

% returned solution is first with minimal residual
if (flag == 0)
    relres = normr_act / n2b;
else
%     r_comp = cellfun(@minus, b, iterapp('mtimes',afun,atype,afcnstr,xmin,varargin{:}), 'uniformoutput', false);
%     if norm_cdcf(r_comp) <= normr_act
%         x = xmin;
%         iter = imin;
%         relres = norm_cdcf(r_comp) / n2b;
%     else
        iter = ii;
        relres = normr_act / n2b;
%     end
end

% truncate the zeros from resvec
if ((flag <= 1) || (flag == 3))
    resvec = resvec(1:ii+1,:);
else
    resvec = resvec(1:ii,:);
end

% only display a message if the output flag is not used
if (nargout < 2)
    itermsg('pcg',tol,maxit,ii,flag,iter,relres);
end
