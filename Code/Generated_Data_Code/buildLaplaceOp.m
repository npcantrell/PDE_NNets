function A = buildLaplaceOp(bounds, n)

% Builds Laplacian operator matrix from CDF approximation
% Assumes symmetric spatial dimensions and periodic boundary conditions

% A = del^2 = (D + X'' + Y'')*(1/delta_x^2);

% CDF = (-4S(m,n) + S(m-1,n) + S(m+1,n) + S(m,n-1) + S(m,n+1))/delta_x^2
    

    % D = -4S(m,n);   
    N = n*n;
    delta_x = (bounds(end) - bounds(1)) / n;
    %Define Diagonals
    v = -4*ones(n*n, 1);
    D = spdiags(v, 0, N, N);

    

    % X'' = S(m-1,n) + S(m+1,n)
    diagonals_inds = [-((n-1)*n) -(n) (n) ((n-1)*n)];
    %Define Diagonals
    v = ones(n*n, 1);
    X_dp = spdiags([v v v v], diagonals_inds, N, N); % allow truncations since vectors are contiunous 1's
    
    % Y'' = S(m,n-1) + S(m,n+1))
    diagonals_inds = [-(n-1) -1 1 (n-1)];
    v_inner = [ones(n-1, 1); 0];
    v_inner_total = repmat(v_inner, n, 1);
    
    v_outer = [1; zeros(n-1, 1)];
    v_outer_total = repmat(v_outer, n, 1);
    
    %Define Diagonals
    V = [v_outer_total v_inner_total flipud(v_inner_total) flipud(v_outer_total)];
    
    Y_dp =  spdiags(V, diagonals_inds, N, N);
    
    % Combine: A = D + X'' + Y''
    A = (1/(delta_x^2)) .* (D + X_dp + Y_dp);

end

