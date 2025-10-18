%% Define Geometry Parameters
ft_to_m = 0.3048; % feet to meter convertion

W   = 0;                 % gutter pan width (m)
W_slope  = 0.04 * ft_to_m ;                % gutter slope (steeper than lane)

l   = 12.0* ft_to_m ;                % lane width (m)
l_slope  = 0.02 *ft_to_m ;             % lane slope = 1/4" per ft

Curb_hight  = 1* ft_to_m ;                    % curb height (m) 

Curb = 8.0 *ft_to_m ;                 % curb width (sidewalk), m
Curb_slope  = 0.02* ft_to_m ;             % curb slope  (same as st)

n_st = 0.013;                   % asphalt
n_curb  = 0.015;     % concrete curb


[A_tbl51, R_tbl51, Y_full] = build_A_R_tables_values( ...
    W, W_slope, l, l_slope, Curb_hight, Curb, Curb_slope, n_st, n_curb);