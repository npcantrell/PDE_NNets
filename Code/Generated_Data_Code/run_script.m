% your solution code goes here
% assign the variables you are asked to save here
% Define spatila and temporal bounds

clear all;

B = 1;
bounds = [-B B];
tspan = linspace(0,20,50);
nu = 0.0001;
n = 50;
num_samples = 1;

% Set Initial Conditions
x = linspace(bounds(1),bounds(2), n+1);
y = linspace(bounds(1), bounds(2), n+1);
[X, Y] = meshgrid(x(1:n), y(1:n));


% Build Derivative Operators
Laplacian = buildLaplaceOp(bounds, n);
[X_der, Y_der] = buildFirstDerOp(bounds, n);

% Define Solutions non-homogeneous term
Laplacian(1,1) = 2;

if ~mod(n, 2)
    kx = (pi/B)*[0 : (n/2 -1) (-n/2) : -1]';
else
    kx = (pi/B)*[0 : (n/2) (-n/2) : -1]';
end    
kx(1) = 10^-6;
ky = kx;

[KX, KY] = meshgrid(kx, ky);
K = zeros(n,n,2);
K(:,:,1) = KX;
K(:,:,2) = KY;

solutions = struct;
solutions.omega = zeros(size(tspan, 2), n^2, num_samples);
solutions.omega_0 = zeros(1, n^2, num_samples);
for i = 1:num_samples
    
    if mod(i, 100) == 0
        disp(i)
    end
    
     amp = 0.5*rand() + 0.5;
     if rand <= 0.5
         x_ellip = (50)*rand() + 75;
         y_ellip = (5)*rand() + 7;
     else
         y_ellip = (50)*rand() + 75;
         x_ellip = (5)*rand() + 7;        
     end
    
    % Test IC
%    amp = 1;
%    x_ellip = 100;
%    y_ellip = 10;
%    
    omega_0 = amp*exp(-1*((X.^2)*x_ellip) -((Y.^2)*y_ellip));

    [t, omega] = ode45(@(t,y) streamfunction_Direct(t, y, Laplacian, X_der, Y_der, nu, K), tspan, omega_0);

    solutions.omega(:,:,i) = omega;
    solutions.omega_0(:,:,i) = reshape(omega_0, 1, n^2);
end
save('SingleGaussNS_20s.mat', '-struct','solutions');
%save('RandomGaussNS_2000_50ts.mat', '-struct','solutions', '-v7.3');
disp('Saved Data:');

