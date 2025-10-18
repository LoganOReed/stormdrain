function RY = R_of_Y(Y, R_tbl, Y_full)

    
    Rt = R_tbl(:)'; 
    R_full=  Rt(end);

    N  = numel(Rt);     
   % clamp Y
    Y = max(0, min(Y, Y_full));

   % cases
    if Y <= 0
        RY = Rt(1);
    elseif Y >= Y_full
        RY = Rt(end);
    else
        % integer portion 
        dy = Y_full / (N - 1);
        k  = floor(Y / dy);
        Yk = k * dy;

        %  linear interpolation
        Rk  = Rt(k + 1);
        Rk1 = Rt(k + 2);
        RY  = Rk + (Y - Yk) * (Rk1 - Rk) / dy;
   end


    RY = RY * R_full;
end
