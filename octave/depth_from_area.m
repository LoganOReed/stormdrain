function Y = depth_from_area(A, A_tbl, Y_full)

    

    At = A_tbl(:)';  % row
    A_full = At(end);
    N = numel(At);  

    if N ~= 51
        error('A_tbl must have length 50 or 51 (got %d).', N);
    end

    At_norm = At / A_full;

    % edge cases 
    a = A / A_full;               
    if a <= 0
        Y = 0; return
    elseif a >= 1
        Y = Y_full; return
    end

    % Bisection to find i with At[i] <= a <= At[i+1] ----
    
    lo = 1;        
    hi = N; 

    while (hi - lo) > 1
        mid = floor((lo + hi)/2);
        if a >= At_norm(mid)
            lo = mid;    
        else
            hi = mid;   
        end
    end


    i = lo;          
    denom = At_norm(i+1) - At_norm(i);
    if denom <= 0
        frac = 0;   % avoid division by zero if plateau
    else
        frac = (a - At_norm(i)) / denom;
    end
    
    Y = (Y_full / (N - 1)) * ( (i - 1) + frac );
end
