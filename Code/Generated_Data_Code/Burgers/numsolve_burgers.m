
clc
clear all
close

% Solving Burgers Equation in 1D: 
% ------------------------------------------
% Authors: Nicholas Cantrell & Ryan Kramlich


% Input Function Parameters 
% x_s < x < x_e
% t_s < t < t_e

% Gaussian IC:
% ------------
% nu = 0.1 
% x grid_size = 200
% t steps = 200
% -10 < x < 10
% 0 < t < 6
% u_left = 0
% u_right = 0 


% Slanted Wave IC:
% -----------------
% nu = 0.018
% x grid_size = 200
% t steps = 200
% -2 < x < 6 
% 0 < t < 1
% u_left = 1
% u_right = 0 


u = solve_burgers(-2, 6, 200, 0, 1, 200, 0.018); 


function u = solve_burgers(x_s, x_e, grid_sz, t_s, t_e, time_stps, viscosity)
    % Boundary Condtions:
    u_left = 1;
    u_right = 0;

    % Other Inital Condition: 
%     u0 = @(x) u_left - (u_left - u_right) .* heaviside(x);
%     u0 = @(x) exp(-(2*(x)).^2);

    % Time and Space Sizes: 
    grid_size = grid_sz; 
    delta_x = (x_e - x_s) / grid_size 
    x = linspace(x_s, x_e, grid_size+1);

    time_steps = time_stps; 
    delta_t = (t_e - t_s) / time_steps
%     delta_t = 0.001;
    tspan = linspace(t_s, t_e, time_steps+1);

    % Initalize Solution and apply IC & BC: 
    u = zeros(grid_size+1, time_steps+1);
    % uncomment for slanted wave IC and remove (x) in line 64 
    u0 = IC(x);
    
    % add u0(x) is using @(x) function IC 
    u(:,1) = u0;
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
        % 3D Plot 
        figure(1)
        
%         [X, T] = meshgrid(x, tspan);
%         surf(X, T, u')
%         xlabel('x'); ylabel('t'); zlabel('u(x,t)')
        
        mesh(x, tspan, u')
        xlabel('x'); ylabel('t'); zlabel('u(x,t)')
        
        
        % 2-D Time stepping animation 
        for time=1:time_steps
            figure(2)
            p = plot(x,u(:,time+1));
            xlim([x_s, x_e])
            ylim([min(u, [], 'all')-0.5, max(u, [] ,'all')+0.5])
            grid on
            xlabel('x')
            ylabel('u(x,t)')
            title(['t = ', num2str(time)])
            drawnow;
            
            if ishghandle(p) == 0
                break;
            end
        end
        
        % Exporting Data
        s1.u = u; 
        s1.x = x;
        s1.t = tspan; 
        
        save('FC_slantedwave_data01.mat', '-struct','s1');
        disp('Saved Data:');
        whos('-file','FC_slantedwave_data01.mat')
    end
end


% Function for piece-wise ICs 
function u0 = IC(x)
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











