function dPsi = psi_prime_from_area(A, A_tbl, R_tbl, Y_full)

    At = A_tbl(:)'; 
    Rt = R_tbl(:)'; 

    % Full area and step size ΔA
    A_full = At(end);
    dA = 0.001 * A_full;

    A = max(0, min(A, A_full));

    % Handle edges: use one-sided diff if we're too close to 0 or A_full
    if A <= dA
        Psi_p = psi_from_area(A + dA, At, Rt, Y_full);
        Psi_0 = psi_from_area(A, At, Rt, Y_full);
        dPsi  = (Psi_p - Psi_0) / dA;               % forward diff
        return
    elseif A >= A_full - dA
        Psi_0 = psi_from_area(A, At, Rt, Y_full);
        Psi_m = psi_from_area(A - dA,  At, Rt, Y_full);
        dPsi  = (Psi_0 - Psi_m) / dA;               % backward diff
        return
    end

    % Central difference in the interior
    Psi_p = psi_from_area(A + dA, At, Rt, Y_full);
    Psi_m = psi_from_area(A - dA, At, Rt, Y_full);
    dPsi  = (Psi_p - Psi_m) / (2*dA);
end
