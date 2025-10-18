load('A_tbl51.mat', 'A_tbl51');   
load('R_tbl51.mat', 'R_tbl51'); 

%% Define Geometry Parameters
ft_to_m = 0.3048; % feet to meter convertion

W   = 0;                 % gutter pan width (m)
W_slope  = 0.04 * ft_to_m ;                % gutter slope (steeper than lane)

l   = 12.0* ft_to_m ;                % lane width (m)
l_slope  = 0.02 *ft_to_m ;             % lane slope = 1/4" per ft

Curb_hight  = 1* ft_to_m ;                    % curb height (m) 

Curb = 8.0 *ft_to_m ;                 % curb width (sidewalk), m
Curb_slope  = 0.02* ft_to_m ;             % curb slope  (same as st)


%% elevations
y_gutter_end = W*W_slope ;
y_crown      = y_gutter_end + l*l_slope;
y_back_edge  = Curb_hight + Curb*Curb_slope;

%% x,y breakpoints from left (curb) to right (crown)
x = [-Curb,  0,   0,     W,      W+l];
y = [y_back_edge, Curb_hight, 0, y_gutter_end, y_crown];
stations = numel(x);

% Shift elevations so minimum y = 0
min_y = min(y);
y = y - min_y;

% Calculate maximum depth
Y_full = max(y);

A_full = A_tbl51(end);

A_target = 0.25 * A_full;    % example: some % of full

%% call functions
Y = depth_Y_from_area(A_target, A_tbl51, Y_full);

R = R_of_Y(Y, R_tbl51, Y_full);

psi = psi_from_area(A_target, A_tbl51, R_tbl51, Y_full);

dPsi = psi_prime_from_area(A_target, A_tbl51, R_tbl51, Y_full);

%% check the results

fprintf('Depth at A = %.6g (%.0f%% of A_full )  is,  Y = %.6f m\n', ...
        A_target, 100*A_target/A_full, Y);

fprintf('Hydraulic radius at Y = %.6f m is, R  = %.6f m\n',Y, R);

fprintf('Section factor at A = %.6g is, Psi  = %.6f m\n',A_target, psi);

fprintf('Derivative of Section factor at A = %.6g is, D_Psi  = %.6f m\n',A_target, dPsi);

