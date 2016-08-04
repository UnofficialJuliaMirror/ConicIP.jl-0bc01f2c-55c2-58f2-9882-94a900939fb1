"""
  imcols(A, b; ϵ = 1e-10)

Removes redundant inequalities in a system of equations

Ax = b

and checks if the equations are consistent.
"""
function imcols(A, b; ϵ = 1e-10)

  # LU Factorization where L has unit diagonals
  if isempty(A)
    return ([], true)
  end

  # presolve - check for consistency
  if norm(A*(A\b) - b, Inf) .> ϵ
    return([], false)
  end

  LUp = lufact(A*A') 
  L = LUp[:L]; U = LUp[:U]; p = LUp[:p]

  # Identify rows in U which are equal to 0, i.e. are linear
  # combinations of the previous rows
  # z = sum(abs(U),2)[:] .< ϵ
  z = abs(diag(U)) .< ϵ

  return (sort(p[!z]), true)

end

"""
ConicIP with preprocessing to ensure the following 
rank constraints

Primal equailty constraints : Gx = d
Rank condition              : rank(G) = size(G,1)

Dual equality constraints   : [ Q A' G'] = c
Rank condition              : rank([Q A' G']) = size(Q,1)
"""
function preprocess_conicIP(Q, c::Matrix, 
  A, b::Matrix, cone_dims, 
  G = spzeros(0,length(c)), d = zeros(0,1); 
  verbose = false,
  options...)

  if verbose == true
    println()
    println(" > INTERIOR POINT PREPROCESSOR v0.7 (July 2016)")
    println()
  end

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  (IP, pconsistent) = imcols(G, d)
  (ID, dconsistent) = imcols([Q A' G[IP,:]'], c)

  if !(pconsistent && dconsistent)
    return ConicIP.Solution(zeros(n,1)/0, zeros(p,1)/0,zeros(m,1)/0, 
      :Infeasible, 0, NaN, NaN, NaN, NaN)
  end

  if (verbose == true) && (length(IP) != p)
    println("   - Removing $(p - length(IP)) redundant primal constraints ");
  end

  if (verbose == true) && (length(ID) != n) 
    println("   - Augmenting $(n - length(ID) ) dual constraints"); 
  end
  
  if (verbose == true) &&  (length(ID) == n) && (length(IP) == p)
    println("   - No changes made")
  end

  z = ones(n); z[ID] = 0; Z = spdiagm(z)

       # Augmented constraints
       #             |
  sol = conicIP(Q + Z, c, A, b, cone_dims, G[IP,:], d[IP,: ]; 
    verbose = verbose,         #                   |
    options...)                # Removed redundant linear constraints 
                               # TODO : (use view?)
                               

  # Argument the dual variables with 0's corresponding to the redundant
  # constraints

  w = zeros(size(G,1),1); w[IP] = sol.w; sol.w = w

  return sol


end