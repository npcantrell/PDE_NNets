function [B, C] = buildFirstDerOp(bounds, n)
    % Build first spatial derivative operator estimate over 2D grid in both
    % directions
    
    N = n*n;
    delta_x = (bounds(end) - bounds(1)) / n;
    
    % B = (S(m+1,n) - S(m-1,n)) / 2*delta_x
    diagonals_inds = [-((n-1)*n) -(n) (n) ((n-1)*n)];
    v = ones(N, 1);
    v2 = -1*ones(N, 1);
    
    %Define Diagonals
    V = [v v2 v v2];
    X_p = spdiags(V, diagonals_inds, N, N);
    
    % C = (S(m,n + 1) - S(m,n - 1)) / 2*delta_x
    
    diagonals_inds = [-(n-1) -1 1 (n-1)];
    
    v_inner = [ones(n-1, 1); 0];
    v_inner_total = repmat(v_inner, n, 1);
    
    v_outer = [1; zeros(n-1, 1)];
    v_outer_total = repmat(v_outer, n, 1);
    
    %Define Diagonals
    V = [v_outer_total -1.*v_inner_total flipud(v_inner_total) -1.*flipud(v_outer_total)];
    
    Y_p =  spdiags(V, diagonals_inds, N, N);
    
    % Multiply by 1/(2*delta_x)
    B = 1/(2*delta_x).*X_p;
    C = 1/(2*delta_x).*Y_p;
end

