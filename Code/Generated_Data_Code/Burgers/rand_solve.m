% Generate Random Gaussian ICs for Burgers Eqn

num_superpositions = 1;
num_samples = 2000;

% For Gaussian IC
%u0 = @(x, y, std, A) A*exp(-((std*(x - y).^2)));

% For Slanted Wave IC
u0 = @IC;

% For Slated Wave Test IC
%u0 = @IC_test;
s1 = struct;

for n = 1:num_samples
    % For Gaussian IC
    %[u, u_0] = solve_burgers(-10, 10, 200, 0, 6, 200, 0.1, u0);
    
    % For Slanted Wave IC
    [u, u_0] = solve_burgers(-2, 6, 200, 0, 1, 200, 0.1, u0);
    
    if n==1
       s1.u = u';
       s1.u_0 = u_0;
    else
        s1.u = cat(3, s1.u, u');
        s1.u_0 = cat(3, s1.u_0, u_0);
    end
    
end

save('Single_SLW.mat', '-struct','s1');
%save('Random_SlantedWave_2000.mat', '-struct','s1');
%save('RandomGauss_2000_reFrame.mat', '-struct','s1');
disp('Saved Data:');
%whos('-file','burgers_data.mat');



function [u, u_0] = solve_burgers(x_s, x_e, grid_sz, t_s, t_e, time_stps, viscosity, u0)
    % Boundary Condtions:
    u_left = 1; % 0 for Guassian
    u_right = 0;

    % Time and Space Sizes: 
    grid_size = grid_sz; 
    delta_x = (x_e - x_s) / grid_size; 
    x = linspace(x_s, x_e, grid_size+1);

    time_steps = time_stps; 
    delta_t = (t_e - t_s) / time_steps;
    tspan = linspace(t_s, t_e, time_steps+1);
    u_0 = u0(x);
    
    %u_0= zeros(1, size(x, 1));
    %u_0 = u0(x, 0, 2, 1);
    
    % For Gaussian IC
    %A = 0.5*rand() + 0.5;
    %std = 2*rand()+ 0.5;
    %u_0 = u_0 + u0(x, 0, std, A);

    % For Slanted Wave IC
    intercept = 3*rand() +0.5;
    u_0 = u_0 + u0(x, intercept);


    % Initalize Solution and apply IC & BC: 
    u = zeros(grid_size+1, time_steps+1);
    
    % add u0(x) is using @(x) function IC 
    u(:,1) = u_0;
    u(1,1) = u_left; 
    u(end, 1) = u_right;
    
    % Parameters:
    nu = viscosity; 
    
    
    % Time Stepping 
    for t = 1:time_steps
        u(1, t+1) = u_left;
        u(end, t+1) = u_right;

        % Applying Finite Difference: 
        for i = 2:grid_size
            u(i, t+1) = u(i,t) - delta_t*(1/(2*delta_x))*(u(i,t) * (u(i+1,t) - u(i-1,t))) ...
                        + nu * delta_t*(1/(delta_x)^2)*(u(i-1,t)-2*u(i,t)+u(i+1, t)); 
        end
    end

    if isnan(norm(u)) == 1
        disp("Something Went Wrong");
    
    else

end


% Function for piece-wise ICs 
function u0 = IC(x, intercept)
    u0 = zeros(size(x));
    for i = 1:length(x)
        if x(i) < 0
            u0(i) = 1;
        elseif x(i) >  intercept
            u0(i) = 0;
        else
            m = (1-0)/ (0 - intercept);
            b = -m*intercept;
            u0(i) = m*x(i) + b;
            %u0(i) = 1 - x(i);
        end
    end
end

% Function for piece-wise ICs 
function u0 = IC_test(x)
    u0 = zeros(size(x));
    for i = 1:length(x)
        if x(i) < 0
            u0(i) = 1;
        elseif x(i) > 1
            u0(i) = 0;
        else
            u0(i) = 1 - x(i);
        end
    end
end










