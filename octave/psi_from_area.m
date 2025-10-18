function Psi = psi_from_area(A, A_tbl, R_tbl, Y_full)

    At = A_tbl(:)'; 
    Rt = R_tbl(:)'; 

    % depth from area
    Y  = depth_Y_from_area(A, At, Y_full);

    %  hydraulic radius at that depth
    RY = R_of_Y(Y, Rt, Y_full);   

    %  section factor
    Psi = A * (RY)^(2/3);
end
