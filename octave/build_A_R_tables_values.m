
function [A_tbl51, R_tbl51, Y_full] = build_A_R_tables_values( ...
    W, W_slope, l, l_slope, Curb_hight, Curb, Curb_slope, n_st, n_curb)

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
  
%%  Manning coeff

n_i  = [n_curb, n_curb, n_st, n_st]; 

%% initial arrays
N_K = 50;                         
A_tbl = zeros(1, N_K);           
R_tbl = zeros(1, N_K);


%% loop
for k = 1:N_K
    % Step 1: depth for the k-th entry
    Y = k * Y_full / 50;


    % Step 1 initialization
    Asum = 0;            % Compound segment area
    Psum = 0;              % Compound wetted perimeter
    Ktot = 0;             % Total flow conductance K 
    i    = 2;            % Transect station index  

   
    while i <= stations
        % Step 2 
        xi_1 = x(i-1);  yi_1 = y(i-1);
        xi   = x(i);    yi   = y(i);

        % Step 3 if Y below both stations, skip this segment
        if Y < min(yi_1, yi)
           
            if (Asum > 0 && Psum > 0)
                Ktot = Ktot + (1.486/n_i(i-1)) * Asum * (Asum/Psum)^(2/3); % Step 9
                Asum = 0; Psum = 0;                                         % Step 9
            end
            i = i + 1;                                                        % Step 10 (move to next station)
            continue
        end

        % Step 4
        w   = xi - xi_1;
        dY  = abs(yi - yi_1);
        p   = sqrt(w^2 + dY^2);

        % Steps 5–6: 
        if Y > max(yi_1, yi)
            % Step 5
            a = w * ( Y - (yi_1 + yi)/2 );
            
        else
            % Step 6
            alpha = (Y - min(yi_1, yi)) / dY;   
            a = (alpha^2) * w * dY;
            w = alpha * w;                      % Step 6 
            p = alpha * p;                      % Step 6 
        end

        % Step 7
        A_tbl(k) = A_tbl(k) + a;
        

        % Step 8
        Asum = Asum + a;
        Psum = Psum + p;

    
        is_end_of_compound = (yi > Y) || (i < stations && n_i(i-1) ~= n_i(i));
        if is_end_of_compound
            if (Asum > 0 && Psum > 0)
                Ktot = Ktot + (1.486/n_i(i-1)) * Asum * (Asum/Psum)^(2/3);  % Step 9
            end
            Asum = 0; Psum = 0;                                             % Step 9
        end

        % Step 10
        i = i + 1;
    end

    % if we ended while still in a compound segment, close it.
    if (Asum > 0 && Psum > 0)
        Ktot = Ktot + (1.486/n_i(end)) * Asum * (Asum/Psum)^(2/3);          % Step 9 
        Asum = 0; Psum = 0;
    end

    % Step 11
    if A_tbl(k) > 0
        R_tbl(k) = ( n_st * Ktot / (1.486 * A_tbl(k)) )^(3/2);           
    else
        R_tbl(k) = 0;
    end
end

fprintf('Computation complete for %d points (k=1,...,50).\n', N_K);
%% save results as .mat

A_tbl51 = [0, A_tbl];    % add k=0 row
R_tbl51 = [0, R_tbl];    

save('A_tbl51.mat','A_tbl51');   % store the 51 points
save('R_tbl51.mat','R_tbl51');        

end