import numpy as np

def newtonBisection(x1, x2, func, tol=1e-6, p=None, maxit=100):
    """
    Using a combination of Newton-Raphson and bisection, find the root of a
    function func bracketed between x1 and x2. The root will be refined until
    its accuracy is known within +/-tol.
    
    Parameters
    ----------
    x1, x2 : float
        Bracketing interval endpoints
    tol : float
        Desired accuracy
    func : callable
        User-supplied function that returns (f, df) where f is the function
        value and df is the derivative. Signature: func(x, p) -> (f, df)
    p : any, optional
        Auxiliary data structure that func may require
    maxit : int, optional
        Maximum number of iterations (default: 100)
    
    Returns
    -------
    rts : float
        The root
    n : int
        Number of function evaluations used, or 0 if maximum iterations exceeded
    
    Notes
    -----
    1. The calling program must ensure that the signs of func(x1) and func(x2)
       are not the same, otherwise x1 and x2 do not bracket the root.
    2. If func(x1) > func(x2) then the order of x1 and x2 should be switched.
    """
    # Check f(low) <= f(high)
    f1, df1 = func(x1)
    f2, df2 = func(x2)
    # Flip endpoints if not appropriate
    if f1 > f2:
        xt = x1
        ft = f1
        dft = df1
        x1 = x2
        f1 = f2
        df1 = df2
        x2 = xt
        f2 = ft
        dft = dft
    
    # Initialize
    x = (x1 + x2) / 2.0  # Initial guess
    xlo = x1
    xhi = x2
    dxold = np.abs(x2 - x1)
    dx = dxold
    
    f, df = func(x, p)
    n = 1
    
    # Loop over allowed iterations
    for j in range(1, maxit + 1):
        # Bisect if Newton out of range or not decreasing fast enough
        if (((x - xhi) * df - f) * ((x - xlo) * df - f) >= 0.0 or
            np.abs(2.0 * f) > np.abs(dxold * df)):
            dxold = dx
            dx = 0.5 * (xhi - xlo)
            x = xlo + dx
            if xlo == x:
                break
        # Newton step acceptable. Take it.
        else:
            dxold = dx
            dx = f / df
            temp = x
            x = x - dx
            if temp == x:
                break
        
        # Convergence criterion
        if np.abs(dx) < tol:
            break
        
        # Evaluate function. Maintain bracket on the root.
        f, df = func(x, p)
        n += 1
        
        if f < 0.0:
            xlo = x
        else:
            xhi = x
    
    rts = x
    if n <= maxit:
        return rts, n
    else:
        return rts, 0

if __name__ == "__main__":
    print("Dont call this :(")
