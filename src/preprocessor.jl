type InconsistentException <: Exception; end

"""
  imcols(A, b; ϵ = 1e-10)

Removes redundant inequalities in a system of equations

Ax = b

and checks if the equations are consistent.
"""
function imcols(A, b; ϵ = 1e-10)


  # LU Factorization where L has unit diagonals
  #(L, U, p) = lu(full(A*A')) 
  if isempty(A)
    return []
  end

  # presolve - check for consistency
  if norm(A*(A\b) - b, Inf) .> ϵ
    throw(InconsistentException())
  end

  LUp = lufact(A*A') 
  L = LUp[:L]; U = LUp[:U]; p = LUp[:p]

  # Identify rows in U which are equal to 0, i.e. are linear
  # combinations of the previous rows
  # z = sum(abs(U),2)[:] .< ϵ
  z = abs(diag(U)) .< ϵ

  return sort(p[!z])

end

"""
  preprocess(Q, c, A, b, G, d; ϵ = 1e-8)

Check for rank deficiencies and correct them, if necessary

Primal equailty constraints : Gx = d
Rank condition              : rank(G) = size(G,1)

Dual equality constraints   : [ Q A' G'] = c
Rank condition              : rank([Q A' G']) = size(Q,1)
"""
function preprocess(Q, c, A, b, G, d; ϵ = 1e-8)

  IP = imcols(G, d)
  ID = imcols([Q A' G[IP,:]'], c)
  return (sort(IP), sort(ID))

end

"""
  intpoint(Q, c, A, b, cone_dims, G, d)

Intpoint with preprocessing for the system

minimize ½yᵀQy - cᵀy s.t Ay >= b Gy = d

cone_dims is an array of tuples (Cone Type, Dimension)

e.g. [("R",2),("Q",4)] means (y₁, y₂) in R+ (y₃, y₄, y₅, y₆) in Q

The preprocessing step removes rank deficiencies in the preconditioned 
matrix 
"""
function preprocess_intpoint(Q, c::Matrix, 
  A, b::Matrix, cone_dims, 
  G = spzeros(0,length(c)), d = zeros(0,1); options...)

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  try
    
    (IP, ID) = preprocess(Q, c, A, b, G, d)

    if length(IP) != p; 
      println("   > Removing $(p - length(IP)) redundant primal constraints ");
    end

    if length(ID) != n; 
      println("   > Augmenting $(n - length(ID) ) dual constraints"); 
    end
    
    z = ones(n); z[ID] = 0; Z = spdiagm(z)

         # Augmented constraints
         #             |
    sol = intpoint(Q + Z, c, A, b, cone_dims, G[IP,:], d[IP,: ]; 
      options...)                #                   |
                                 # Removed redundant linear constraints 
                                 # TODO : (use view?)

    # Argument the dual variables with 0's corresponding to the redundant
    # constraints

    w = zeros(size(G,1),1); w[IP] = sol.w; sol.w = w
    return sol

  catch e
    
    # The preprocessor might throw an exception if the system of linear
    # equalities is inconsistent, i.e. the QP is infeasible.

    if isa(e, InconsistentException) 
      return IntPoint.Solution(zeros(n,1)/0, 
        zeros(p,1)/0, 
        zeros(m,1)/0, 
        :Infeasible, 0, NaN, NaN, NaN, NaN)
    else 
      throw(e); 
    end

  end

end