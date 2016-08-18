importall MathProgBase.SolverInterface
import Base.convert

# --------------------------------------------------------------------------------
# Begin implementation of the MPB low-level interface
# Implements
# - ConicModel
# - loadproblem!
# - optimize!
# - status
# - numvar
# - numconstr
# http://mathprogbasejl.readthedocs.org/en/latest/solverinterface.html
#

immutable ConicIPSolver <: AbstractMathProgSolver
  preprocess
  options
end

ConicIPSolver(;equalities_as_double_inequalities = false,
                preprocess = true,
                kwargs...) = 
                ConicIPSolver( preprocess, 
                                kwargs )

"""
Type which encapsulates a ConicIP model, and a way to convert
its output into a MPB model

minimize    ½yᵀQy - cᵀy
s.t         Ay >= b
            Gy  = d
"""
type ConicIPModel <: AbstractConicModel

    # Parameters to be passed into ConicIP

    Q           ::AbstractMatrix
    c           ::AbstractMatrix
    A           ::AbstractMatrix
    b           ::AbstractMatrix
    cone_dims  
    G           ::AbstractMatrix
    d           ::AbstractMatrix
 
    # Solutions

    solve_stat 
    sol 
    obj_val     ::Float64
    primal_sol  ::Vector{Float64}
    dual_sol    ::Vector{Float64}
    vardual_sol ::Vector{Float64}
    slack       ::Float64

    # Bookkeeping for indicies, so we can figure out what
    # the dual variables are.

    I_Al        ::Vector{Int64}
    I_Gl        ::Vector{Int64}
    I_A         ::Vector{Int64}
    I_G         ::Vector{Int64}

    I_vAl       ::Vector{Int64}
    I_vGl       ::Vector{Int64}
    I_vA        ::Vector{Int64}
    I_vG        ::Vector{Int64}

    # Number of constraints and varconstraints
    n_constr
    n_varconstr

    preprocess  ::Bool
    options

    nA          ::Float64

end

"""
  signed_assign!(x, y, I, J)

signed assignment, a hack to pass in sign information through 
indices.

x[I] =  y[J]  when J .> 0
x[I] = -y[-J] when J .< 0
"""
function signed_assign!(x, y, I, J)
  I = abs(I)
  P = J .> 0
  x[I[P]] = y[J[P]]
  x[I[!P]] = -y[-J[!P]]
end

ConicModel(s::ConicIPSolver) = ConicIPModel(
  zeros(0,0), 
  zeros(0,0), 
  zeros(0,0), 
  zeros(0,0), 
  zeros(0,0), 
  zeros(0,0), 
  zeros(0,0),
  nothing, 
  nothing, 
  0, 
  zeros(0),
  zeros(0), 
  zeros(0),   
  0,
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  round(Int, zeros(0)),
  0,
  0,
  s.preprocess,
  s.options,
  NaN)

LinearQuadraticModel(s::ConicIPSolver)  = ConicToLPQPBridge(ConicModel(s))
status(m::ConicIPModel)                = m.solve_stat
getobjval(m::ConicIPModel)             = m.obj_val
getsolution(m::ConicIPModel)           = copy(m.primal_sol)
numvar(m::ConicIPModel)                = length(m.c)
numconstr(m::ConicIPModel)             = m.n_constr + m.n_varconstr
supportedcones(s::ConicIPSolver)        = [:Free, 
                                           :Zero, 
                                           :NonNeg, 
                                           :NonPos, 
                                           :SOC,
                                           :SDP]

function show(m::ConicIPModel)

  println("Q = ",full(m.Q))
  println("c = ",m.c)
  println("A = ",full(m.A))
  println("b = ",m.b)
  println("cone_dims = ", m.cone_dims)
  println("G = ",full(m.G))
  println("d = ",m.d)
  
end

function loadproblem!(m::ConicIPModel, c, 
  A_, b_, constr_cones, var_cones)

  n = size(A_, 2)

  A_ = sparse(A_)

  A = spzeros(0,n)
  b = zeros(0,1)
  G = spzeros(0,n)
  d = zeros(0,1)

  I_A   = round(Int, zeros(0))
  I_G   = round(Int, zeros(0))
  I_Al  = round(Int, zeros(0))
  I_Gl  = round(Int, zeros(0))

  I_vA  = round(Int, zeros(0))
  I_vG  = round(Int, zeros(0))
  I_vAl = round(Int, zeros(0))
  I_vGl = round(Int, zeros(0))

  n_constr    = sum([length(ind) for (cone, ind) = constr_cones])
  n_varconstr = sum([length(ind) for (cone, ind) = var_cones])

  cone_dims = []

  Gs = 0
  As = 0

  for (cone, ind) = constr_cones

    if typeof(ind) <: Integer
      ind = ind:ind
    end

    if cone == :Zero
      i = Gs; Gs = Gs + length(ind); j = Gs
      d = [ d ; b_[ind,:] ]
      append!( I_G  , ind   )
      append!( I_Gl , i+1:j )
    end

    if cone == :NonPos
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("R", length(ind)) )
      b = [b ; b_[ind,:] ]
      append!( I_A  , ind      )
      append!( I_Al , -(i+1:j) )
    end

    if cone == :NonNeg
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("R", length(ind)) )
      b = [ b ; -b_[ind,:] ]
      append!( I_A  , -ind   )
      append!( I_Al , i+1:j )
    end

    if cone == :SOC
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("Q", length(ind)) )
      b = [ b ; -b_[ind,:] ]
      append!( I_A  , -ind   )
      append!( I_Al , i+1:j )
    end

    if cone == :SDP
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("S", length(ind)) )
      b = [ b ; -b_[ind,:] ]
      append!( I_A  , -ind   )
      append!( I_Al , i+1:j )
    end

  end

  I = speye(n,n)

  cum_free_dim = 0

  base_sise_A = size(A,1)
  base_size_G = size(G,1)

  for (cone, ind) = var_cones

    if typeof(ind) <: Integer
      ind = ind:ind
    end

    if cone == :Free
      cum_free_dim = cum_free_dim + length(ind)
    end

    if cone == :Zero
      i = Gs; Gs = Gs + length(ind); j = Gs
      d = [d ; zeros(length(ind), 1) ]
      append!( I_vG  , ind     )
      append!( I_vGl , (i+1:j) ) 
    end

    if cone == :NonPos
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("R", length(ind)) )
      b = [ b ; zeros(length(ind), 1) ]
      append!( I_vA  , -ind      )
      append!( I_vAl , -(i+1:j) )
    end

    if cone == :NonNeg
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("R", length(ind)) )
      b = [ b ; zeros(length(ind), 1) ]
      append!( I_vA  , ind     )
      append!( I_vAl , (i+1:j) ) 
    end

    if cone == :SOC
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("Q", length(ind)) )
      b = [ b ; zeros(length(ind), 1) ]
      append!( I_vA  , ind     )
      append!( I_vAl , (i+1:j) )  
    end

    if cone == :SDP
      i = As; As = As + length(ind); j = As
      push!( cone_dims, ("S", length(ind)) )
      b = [ b ; zeros(length(ind), 1) ]
      append!( I_vA  , ind     )
      append!( I_vAl , (i+1:j) )  
    end

  end
  
  nA = vecnorm(A_)

  Λ = sign([I_A;I_vA])
  A = Λ.*[A_[abs(I_A),:]; nA*I[abs(I_vA),:]]

  Λ = sign([I_G;I_vG])
  G = Λ.*[A_[abs(I_G),:]; nA*I[abs(I_vG),:]]

  m.Q           = spzeros(n,n)
  m.c           = -c''
  m.A           = A
  m.b           = b
  m.cone_dims   = cone_dims
  m.G           = G
  m.d           = d
  m.I_A         = I_A
  m.I_Al        = I_Al
  m.I_G         = I_G
  m.I_Gl        = I_Gl
  m.I_vA        = I_vA
  m.I_vAl       = I_vAl
  m.I_vG        = I_vG
  m.I_vGl       = I_vGl
  m.n_constr    = n_constr
  m.n_varconstr = n_varconstr
  m.nA          = nA

end

function optimize!(m::ConicIPModel)

  if m.preprocess
    m.sol = preprocess_conicIP(m.Q,m.c,m.A,m.b,m.cone_dims,m.G,m.d; m.options...)
  else
    m.sol = conicIP(m.Q,m.c,m.A,m.b,m.cone_dims,m.G,m.d; m.options...)
  end

  m.solve_stat = m.sol.status

  m.obj_val = -vecdot(m.sol.y,m.c)
  m.primal_sol = m.sol.y[:]

  m.dual_sol = zeros(m.n_constr)
  signed_assign!(m.dual_sol, m.sol.v, m.I_A, m.I_Al)
  signed_assign!(m.dual_sol, m.sol.w, m.I_G, m.I_Gl)

  m.vardual_sol = zeros(m.n_varconstr)
  signed_assign!(m.vardual_sol, m.sol.v*m.nA, m.I_vA, m.I_vAl)
  signed_assign!(m.vardual_sol, m.sol.w*m.nA, m.I_vG, m.I_vGl)

end

getdual(m::ConicIPModel)    = m.dual_sol
getvardual(m::ConicIPModel) = m.vardual_sol