%  one-sided street cross-section figure 

ft_to_m = 0.3048; % feet to meter convertion

W   = 0;                 % gutter pan width (m)
W_slope  = 0.04 * ft_to_m ;                % gutter slope (steeper than lane)

l   = 12.0* ft_to_m ;                % lane width (ft)
l_slope  = 0.02 *ft_to_m ;             % lane slope = 1/4" per ft

Curb_hight  = 1* ft_to_m ;                    % curb height (m) 

Curb = 8.0 *ft_to_m ;                 % curb width (sidewalk), m
Curb_slope  = 0.02* ft_to_m ;             % curb slope  (same as st)


% elevations
y_gutter_end = W*W_slope ;
y_crown      = y_gutter_end + l*l_slope;
y_back_edge  = Curb_hight + Curb*Curb_slope;

% x,y breakpoints from left (back-of-curb) to right (crown)
x = [-Curb,  0,   0,     W,      W+l];
y = [y_back_edge, Curb_hight, 0, y_gutter_end, y_crown];

% plot
figure('Color','w'); hold on
plot(x, y, 'b-', 'LineWidth', 3);  
axis equal
xlabel('x (m)'); ylabel('elevation y (m)');
title('One-sided Street Cross-Section ');
grid on; box on; 
xticks(min(x):0.5:max(x));     
