function domega_dt = streamfunction_Direct(t, omega, Laplacian, X_der, Y_der, nu, K, L, U, P)

% 1) Solve for Psi

    % Run LU decomp if L matrix exists
    if exist('L', 'var')
        y = L\(P*omega); 
        psi = U \ y ;
        
    % Run Spectral solve if K indices exist and is not empty
    elseif exist('K', 'var') && ~isempty(K)
        omega_hat = fft2(reshape(omega, size(K, 1), size(K, 1))); 
        F_psi = -1 * omega_hat ./ (K(:,:,1).^2 + K(:,:,2).^2);      
        psi = ifft2(F_psi);
        psi = real(reshape(psi, size(K, 1)^2, 1));
    
    % Solve Directly with GE    
    else
        psi = Laplacian\omega;
    end

% 2) Solve Stream Function
    domega_dt = nu.*(Laplacian*omega) - (X_der*psi).*(Y_der*omega) + (Y_der*psi).*(X_der*omega);
end

