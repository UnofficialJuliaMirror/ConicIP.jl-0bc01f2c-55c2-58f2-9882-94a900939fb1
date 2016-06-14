isdefined(Base, :__precompile__) && __precompile__()

module IntPoint

export Id, Diag, intpoint, pivot3gen

import Base:+,*,-,\,^
using Base.LinAlg.BLAS:axpy!,scal!
using SymWoodburyMatrices
using BlockMatrices

ViewTypes   = Union{SubArray}
VectorTypes = Union{Matrix, Vector, ViewTypes}
MatrixTypes = Union{Matrix, Array{Real,2},
                    SparseMatrixCSC{Real,Integer}}

# returns 0 for matrices with dimension 0.
normsafe(x) = isempty(x) ? 0 : norm(x)

# ──────────────────────────────────────────────────────────────
#  3x1 block vector
# ──────────────────────────────────────────────────────────────

type v4x1; y::Matrix; w::Matrix; v::Matrix; s::Matrix; end

+(a::v4x1, b::v4x1) = v4x1(a.y + b.y, a.w + b.w, a.v + b.v, a.s + b.s)
*(α::Real, a::v4x1) = v4x1(α*a.y, α*a.w, α*a.v, α*a.s);
-(a::v4x1)          = v4x1(-a.y, -a.w, -a.v, -a.s);
-(a::v4x1, b::v4x1) = v4x1(a.y - b.y, a.w - b.w, a.v - b.v, a.s - b.s)
Base.norm(a::v4x1)  = norm(a.y) + normsafe(a.w) + normsafe(a.v) + normsafe(a.s) 
Base.println(z::v4x1)  = println("y: ", z.y',
                         "\ng: ", z.w,
                         "\nv: ", z.v',
                         "\ns: ", z.s');

function axpy4!(α::Number, x::v4x1, y::v4x1)
    axpy!(α, x.y, y.y); axpy!(α, x.w, y.w)
    axpy!(α, x.v, y.v); axpy!(α, x.s, y.s)
end

# ──────────────────────────────────────────────────────────────
#  Linear operator representing a congurance transform of a
#  matrix in vectorized form
# ──────────────────────────────────────────────────────────────

type VecCongurance; R :: Matrix; end

*(W::VecCongurance, x::VectorTypes)    = vec(W.R'*mat(x)*W.R)
\(W::VecCongurance, x::VectorTypes)    = vec(inv(W.R)'*mat(x)*inv(W.R))
Base.inv(W::VecCongurance)             = VecCongurance(inv(W.R))
^(W::VecCongurance, n::Integer)        = VecCongurance(W.R^n)
Base.size(W::VecCongurance, i)         = int(size(W.R,1)*(size(W.R,1)+1)/2)

function Base.full(W::VecCongurance)
  n = size(W,1)
  I = eye(n)
  Z = zeros(n,n)
  for i = 1:n
    Z[:,i] = W*I[:,i][:]
  end
  return Z
end

ord(x) = begin; n = length(x); int((sqrt(1+8*n) - 1)/2); end

function mat(x)

  # inverse of vec
  # > mat([1,2,3,4,5,6)]
  #  1    2/√2  3√2
  #  2    4     5√2
  #  3√2  5√2   6

  n = ord(x)
  Z = zeros(n,n)
  for i = 1:n
    k = length(x) - (n-i+2)*(n-i+1)/2
    for j = 1:n
      if i <= j
        if i == j
          Z[i,j] = x[k+j-i+1]
        else
          Z[i,j] = x[k+j-i+1]/√2
        end
      else
        Z[i,j] = Z[j,i];
      end
    end
  end
  return Z

end

function vec(Z)

  # inverse of mat
  # > vec([1 2 3; 2 4 5; 3 5 6])
  # [1 2√2 3√2 4 5√2 6]

  n = size(Z,1)
  x = zeros(int(n*(n+1)/2),1)
  c = 1
  for i = 1:n
    for j = 1:n
      if i <= j
        if i == j
          x[c] = Z[i,j];
        else
          x[c] = Z[i,j]*√2;
        end
        c = c + 1;
      end
    end
  end
  return x

end

# ──────────────────────────────────────────────────────────────
#  Misc Helper Types/Functions
# ──────────────────────────────────────────────────────────────

# Example:  block_ranges([1,1,3]) = [1:1,2:2,3:5]
cum_range(x) = [i:(j-1) for (i,j) in
        zip(cumsum([1;x])[1:end-1], cumsum([1;x])[2:end])]
QF(z::Vector)                    = 2*z[1]*z[1] - dot(z,z) # xᵀJx
Q(x::VectorTypes,y::VectorTypes) = 2*x[1]*y[1] - dot(x,y) # xᵀJy
fts(x₁, α₁, y₁, x₂, α₂, y₂)      = (x₁'*x₂) - α₂*(x₁'*y₂) -
          α₁*(y₁'*x₂) + α₁*α₂*(y₁'*y₂) # (x₁ - α₁*y₁)'(x₂ - α₂y₂)

function nestod_soc(z,s)

  # Nesterov-Todd Scaling Matrix for the second order cone
  # Matrix which satisfies the properties
  # W*z = inv(W)*s

  QF = r -> 2*r[1]*r[1] - dot(r,r)

  n = size(z,1)

  β = (QF(s)/QF(z))^(1/4)

  # Normalize z,s vectors
  z = z/sqrt(QF(z))
  s = s/sqrt(QF(s))

  γ = sqrt((1 + (z'*s)[1])/2)

  # Jz = J*z;
  Jz = -z; Jz[1] = -Jz[1]

  w = (1./(2.*γ))*(s + Jz)
  w[1] = w[1] + 1
  v = (sqrt(2*β)/sqrt(2*w[1]))*w

  J = Diag(β*ones(n))
  J.diag[1] = -β

  return SymWoodbury(J, reshape(v, length(v), 1), ones(1,1))

end


function nestod_sdc(z,s)

  # Nesterov-Todd Scaling Matrix for the Semidefinite Cone
  # Matrix which satisfies the properties
  # W*z = inv(W)*sb
  Z  = mat(z); S  = mat(s); Sq = S^(0.5)
  #return VecCongurance((Sq*((Sq*Z*Sq)^(-0.5))*Sq)^(0.5))
  return VecCongurance(S^(0.125)*Z^(-0.25)*S^(0.125))

end

function maxstep_rp(x,d)

  # Assume x in R+.
  # Returns maximum α such that x + α*d in R+.

  # assert(all(x .> 0));

  #I = d .>= 0;
  #if all(!I); return 1; end # all directions unbounded

  minVal = Inf
  for i = 1:length(x)
    if d[i] > 0
      minVal = min(minVal, x[i]/d[i])
    end
  end
  return minVal

end

function maxstep_rp(x, e::Void)

  # Let α = inf { α | -x + αe >= 0 }
  # Then this returns
  # 0       if α < 0   (point is STRICTLY feasible)
  # 1 + α   otherwise

  if all(x .> 0)
    return 0;
  else
    return -1 + minimum(x);
  end

end

function maxstep_soc(x,d)

  # Assume x in Q.
  # Returns maximum α such that x - α*d in Q.

  # assert((norm(x[2:end]) <= x[1]))

  # J = -speye(size(x,1)); J[1,1] = 1; # Hyperbolic Identiy

  d = -d;
  γ = Q(x,x)
  xbar = x/sqrt(γ)
  β = Q(xbar,d)

  ρ1 = β /sqrt(γ)
  μ  = ((β + d[1])/(xbar[1] + 1))[1]
  ρ2 = (d[2:end] - μ*xbar[2:end])
  alpha = norm(ρ2)/sqrt(γ) - ρ1
  if alpha < 0
    return Inf
  else
    return 1/alpha
  end

end

function maxstep_soc(x, e::Void)

  # Maximum step to cone
  α = norm(x[2:end]) - x[1];
  return α < 0 ? 0 : -1 - α;

end

function maxstep_sdc(x,d)

  # Maximum step to Semidefinite cone
  X     = mat(x)^(-1/2);
  D     = mat(d)
  XDX   = X*D*X
  XDX   = 0.5*(XDX + XDX')
  (Λ,_) = eig(XDX)
  Λn    = Λ .< 0
  if all(Λn)
    return Inf
  else
    return 1/maximum(Λ[!Λn])
  end

end

function maxstep_sdc(x,d::Void)

  # Maximum step to Semidefinite cone
  X = mat(x)
  (Λ,_) = eig(X)
  minΛ  = minimum(Λ)
  return all(minΛ .> 0) ? 0 : -1 + minΛ

end

function drp!(x, y, o)

    for i = 1:length(x); o[i] = x[i]/y[i]; end

end

function xrp!(x, y, o)

    for i = 1:length(x); o[i] = x[i]*y[i]; end

end

function dsoc!(y,x, o)

  # Inverse of arrow matrix
  #     ┌                         ┐ ┌    ┐
  # α⁻¹ │  y1  -yb                │ │ x1 │
  #     │ -yb   (αI + yb*yb')/y1  │ │ xb │
  #     └                         ┘ └    ┘

  y1 = x[1]; yb = x[2:end]
  α = y1^2 - vecdot(yb,yb)

  x1 = y[1]; xb = y[2:end]
  o[1] = (y1*x1 - vecdot(yb,xb) )/α
  o[2:end] = (-yb*x1 + (α*xb + vecdot(yb,xb)*yb)/y1)/α

end

function xsoc!(x, y, o)

  o[1] = dot(x,y)
  for i = 2:length(x); o[i] = x[1]*y[i] + y[1]*x[i]; end

end

function dsdc!(x, y, o)

  n = int(sqrt(size(x,1)))
  X = mat(x); Y = mat(y)
  o[:] = vec(lyap(Y,-X))

end

function xsdc!(x, y, o)

  X = mat(x); Y = mat(y)
  o[:] = vec(X*Y + Y*X)

end

# ──────────────────────────────────────────────────────────────
#  KKT Solvers
# ──────────────────────────────────────────────────────────────

"""
Solves the 3x3 system

┌             ┐ ┌    ┐   ┌   ┐
│ Q   G'  -A' │ │ y' │ = │ y │
│ G           │ │ w' │   │ w │ 
│ A        F² │ │ v' │   │ v │
└             ┘ └    ┘   └   ┘
"""
function solve3x3gen_sparse_dense(F, F⁻¹, Q, A, G)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  F² = sparse(F^2)
  Q  = sparse(Q) # TODO: remove the need for this
  A  = sparse(A)
  G  = sparse(G)

  Z = [ Q        G'            -A' 
        G        spzeros(p,p)   spzeros(p,m)
        A        spzeros(m,p)   F²            ]

  ZZ = lufact(Z)

  function solve3x3(Δy, Δw, Δv)

    z = ZZ\[Δy; Δw; Δv]
    return (z[1:n,:], z[n+1:n+p,:], z[n+p+1:end,:])

  end

  return solve3x3

end

function lift(F::Block)

  d = zeros(0)

  IA, JA, VA = Int[], Int[], Float64[]
  IB, JB, VB = Int[], Int[], Float64[]
  ID, JD, VD = Int[], Int[], Float64[]

  n = BlockMatrices.block_idx(F)[end][end]
  Ir = 0   # Index of top right coordinate for expansion

  for (In,Blk) = zip(BlockMatrices.block_idx(F), F.Blocks)

    if isa(Blk, SymWoodbury)

      for i = 1:length(Blk.A.diag)
        push!(IA,In[i]); push!(JA,In[i]); push!(VA,Blk.A.diag[i])
      end

      for i = 1:size(Blk.B,1), j = 1:size(Blk.B,2)
        push!(IB,In[i]); push!(JB,Ir+j); push!(VB,Blk.B[i,j])
      end

      for i = 1:size(Blk.D,1), j = 1:size(Blk.D,2)
        if Blk.D[i,j] != 0
          push!(ID,Ir+i); push!(JD,Ir+j); push!(VD,-1/Blk.D[i,j])
        end
      end

      Ir = Ir + size(Blk.B,2)

    end

    if isa(Blk, Diag)

      for i = 1:length(Blk.diag)
        push!(IA, In[i]); push!(JA, In[i]); push!(VA, Blk.diag[i])
      end

    end

  end
  return (sparse(IA,JA,VA), sparse(IB,JB,VB,n,Ir), sparse(ID,JD,VD));

end

function solve3x3gen_sparse_lift(F, F⁻¹, Q, A, G)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  (F²A, F²B, invF²D) = lift(F^2)

  F² = sparse(F^2)
  Q  = sparse(Q) # TODO: remove the need for this
  A  = sparse(A)
  G  = sparse(G)
  r  = size(invF²D,1)

  Z = [ Q             G'            -A'            spzeros(n,r)
        G             spzeros(p,p)   spzeros(p,m)  spzeros(p,r)
        A             spzeros(m,p)   F²A           F²B                         
        spzeros(r,n)  spzeros(r,p)   F²B'          invF²D        ]

  ZZ = lufact(Z)

  function solve3x3(Δy, Δw, Δv)

    z = ZZ\[Δy; Δw; Δv; zeros(r,1)]
    return (z[1:n,:], z[n+1:n+p,:], z[(n+p+1):(n+m+p),:])

  end

  return solve3x3

end

function solve3x3gen_sparse(F, F⁻¹, Q, A, G)

  function count_lift(F)
    n = 0
    for Blk = F.Blocks
      if isa(Blk, SymWoodbury)
        n = n + size(Blk,1) + 2*length(Blk.B) + 3
      end
      if isa(Blk, Diag)
        n = n + size(Blk,1)
      end
    end
    return n
  end


  function count_dense(F)
    n = 0
    for Blk = F.Blocks
      if isa(Blk, SymWoodbury)
        n = n + size(Blk,1)^2
      end
      if isa(Blk, Diag)
        n = n + size(Blk,1)
      end
    end
    return n
  end

  if count_lift(F) < count_dense(F)
    return solve3x3gen_sparse_lift(F, F⁻¹, Q, A, G)
  else 
    return solve3x3gen_sparse_dense(F, F⁻¹, Q, A, G)
  end    
end

"""
Solves the 2x2 system

┌                ┐ ┌    ┐   ┌   ┐
│ Q + A'F⁻²A  G' │ │ y' │ = │ y │
│ G              │ │ w' │   │ w │ 
└                ┘ └    ┘   └   ┘
"""
function solve2x2gen(F, F⁻¹, Q, A, G)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  F⁻² = sparse(inv(F^2)) # TODO: Think about how it might help to pass in F^-1 

  Z = [ Q + A'*F⁻²*A   G'            
        G              spzeros(p,p) ]

  Z = lufact(Z)

  function solve2x2(Δy, Δw)

    z = Z\[Δy; Δw]
    return (z[1:n,:], z[n+1:end,:])

  end

  return solve2x2

end

"""
Wrapper around solve2xegen to solve 3x3 systems by pivoting
on the third component.
"""
function pivot3(solve2x2gen, F, F⁻¹, Q, A, G)

  F⁻² = F⁻¹*F⁻¹
  solve2x2 = solve2x2gen(F, F⁻¹, Q, A, G)

  function solve3x3(y, w, v)

    t1 = F⁻²*v
    (Δy, Δw) = solve2x2(y + A'*t1, w)
    axpy!(-1, F⁻²*(A*Δy), t1)  # Δv = F⁻²*(v - A*Δy)

    return(Δy, Δw, t1)

  end

end

pivot3gen(solve2x2gen) = (F,F⁻¹,Q,A,G) -> pivot3(solve2x2gen,F,F⁻¹,Q,A,G)

# ──────────────────────────────────────────────────────────────
#  Interior Point
# ──────────────────────────────────────────────────────────────

type Solution

  y      :: Matrix  # primal
  w      :: Matrix  # dual (linear equality)
  v      :: Matrix  # dual (linear inequality)
  status :: Symbol  # :Optimal, :Infeasible
  Iter   :: Integer # number of iterations
  Mu     :: Real    # optimality conditions
  prFeas :: Real
  duFeas :: Real
  muFeas :: Real

end

"""
  intpoint(Q, c, A, b, cone_dims, G, d)

Interior point solver for the system

minimize    ½yᵀQy - cᵀy
s.t         Ay >= b
            Gy  = d

cone_dims is an array of tuples (Cone Type, Dimension)

e.g. [("R",2),("Q",4)] means
(y₁, y₂)          in  R+
(y₃, y₄, y₅, y₆)  in  Q

SDP Cones are NOT supported and purely experimental at this
point.
"""
function intpoint(

  # ½xᵀQx - cᵀx
  Q, c::Matrix,

  # Ax ≧ b
  A, b::Matrix, cone_dims,

  # Gx = d
  G = spzeros(0,length(c)), d = zeros(0,1);

  # Solver Parameters

  # L = solve3x3gen(F,Q,A,G)
  # L(a,b,c) solves the system
  # ┌             ┐ ┌   ┐   ┌   ┐
  # │ Q   G'  -A' │ │ a │ = │ y │
  # │ G           │ │ b │   │ w │ 
  # │ A        F² │ │ c │   │ v │
  # └             ┘ └   ┘   └   ┘  
  #
  # We can also wrap a 2x2 solver using pivot3gen(solve2x2gen)
  # The 2x2 solves the system
  # 
  # L = solve2x2gen(F,Q,A,G)
  # L(a,b) solves
  # ┌                ┐ ┌   ┐   ┌   ┐
  # │ Q + AᵀF²A   G' │ │ a │ = │ y │
  # │ G              │ │ b │   │ w │
  # └                ┘ └   ┘   └   ┘
  solve3x3gen = solve3x3gen_sparse,

  optTol = 1e-5,           # Optimal Tolerance
  DTB = 0.01,              # Distance to Boundary
  verbose = true,          # Verbose Output
  maxRefinementSteps = 3,  # Maximum number of IR Steps
  maxIters = 100,          # Maximum number of interior iterations
  cache_nestodd = false,   # Set to true if there are many small blocks
  refinementThreshold = optTol/1e7 # Accuracy of refinement steps

  )

  # Precomputed transposition matrices
  Aᵀ = A'; Gᵀ = G'

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  block_types  = [i[1] for i in cone_dims]
  block_sizes  = [i[2] for i in cone_dims]
  block_data   = zip(block_types, cum_range(block_sizes),
                     [i for i in 1:length(block_types)])

  # Sanity Checks
  ◂ = nothing
  #!(m == 0 && length(block_sizes) == 0) || m != sum(block_sizes) ? error("Inconsistency in inequalities") : ◂
  size(Q,1) != size(Q,2)? error("Q is not square") : ◂
  size(b,1) != m        ? error("Inconsistency in inequalities") : ◂
  size(c,1) != n        ? error("Inconsistency in inequalities/objective") : ◂
  size(d,1) != p        ? error("Inconsistency in equalities") : ◂
  size(G,2) != n        ? error("Inconsistency in equalities/objective") : ◂

  # Number to scale (z's) by
  # 1 for each R_+ dimension
  # 1 for each Q cone (regardless of dimension)
  conedim = 0
  for (btype, I, i) = block_data
    if btype == "R"; conedim += length(I);  end
    if btype == "Q"; conedim += 1;          end
    if btype == "S"; conedim += ord(I);     end
  end

  # e = Group identity
  # Concatenate the vectors
  # [1, 1, … , 1] for R_+
  # [1, 0, … , 0] for Q

  e = zeros(m,1)
  for (btype, I, i) = block_data
    m_i = length(I)
    if btype == "R"; e[I] = ones(m_i,1);             end
    if btype == "Q"; e[I] = [1; zeros(m_i-1,1)];     end
    if btype == "S"; e[I] = vec(eye(ord(I)));        end
  end

  # ──────────────────────────────────────────────────────────────
  #  Functions capturing cone_dims
  # ──────────────────────────────────────────────────────────────

  function maxstep(x, d)

    # Linesearch

    min_α = Inf;
    for (btype, I, i) = block_data
      xI = sub(x,I)
      dI = ( d == nothing ? nothing : sub(d,I) )
      if btype == "R"; α = maxstep_rp(xI,dI);  end
      if btype == "Q"; α = maxstep_soc(xI,dI); end
      if btype == "S"; α = maxstep_sdc(xI,dI); end
      min_α = min(α, min_α)
    end

    return min_α;

  end

  function is_feas(x)

    # Check if x is feasible

    for (btype, I, i) = block_data
      xI = sub(x,I); yI = sub(y,I);
      if btype == "R"; assert(x[I] > 0);                    end
      if btype == "Q"; assert(norm(x[I][2:end]) - x[I][1]); end
    end

  end

  function nt_scaling(x, y)

    # Compute Nesterov-Todd scaling matrix, F s.t.
    # λ = F*x = F\y

    B = Block(size(block_sizes,1));

    for (btype, I, i) = block_data
      xI = sub(x,I); yI = sub(y,I);
      if btype == "R"; B[i] = Diag(sqrt(yI./xI)); end
      if btype == "Q"; B[i] = nestod_soc(xI, yI); end
      if btype == "S"; B[i] = nestod_sdc(xI, yI); end
    end

    return B;

  end

  function ÷(x,y)

    # Group division x ○\ y

    o = zeros(length(x),1)
    for (btype, I, i) = block_data
      xI = sub(x,I); yI = sub(y,I); oI = sub(o,I)
      if btype == "R"; drp!(xI, yI, oI);  end
      if btype == "Q"; dsoc!(xI, yI, oI); end
      if btype == "S"; dsdc!(xI, yI, oI); end
    end
    return o;

  end

  function ∘(x,y)

    # Group product x ○ y

    o = zeros(length(x),1)
    for (btype, I, i) = block_data
      xI = sub(x,I); yI = sub(y,I); oI = sub(o,I)
      if btype == "R"; xrp!(xI, yI, oI);  end
      if btype == "Q"; xsoc!(xI, yI, oI); end
      if btype == "S"; xsdc!(xI, yI, oI); end
    end
    return o;

  end

  function solve4x4gen(λ, F, F⁻¹, solve3x3gen = solve3x3gen)

    #
    # solve4x4gen(λ, F)(r) solves the 4x4 KKT System
    # ┌                  ┐ ┌    ┐   ┌     ┐
    # │ Q   G'  -A'      │ │ Δy │ = │ r.y │
    # │ G                │ │ Δw │   │ r.w │ S = block(λ)*F
    # │ A             -I │ │ Δv │   │ r.v │ V = block(λ)*inv(F)
    # │          S     V │ │ Δs │   │ r.s │
    # └                  ┘ └    ┘   └     ┘
    # F = Nesterov-Todd scaling matrix
    #

    solve3x3 = solve3x3gen(F, F⁻¹, Q, A, G)

    function solve4x4(r)
      
      t1 = F*(r.s ÷ λ) 
      (Δy, Δw, Δv)  = solve3x3(r.y, r.w, r.v + t1)
      axpy!(-1, F*(F*Δv), t1) # > Δs = t1 - F*(F*Δv)
      return v4x1(Δy,Δw,Δv,t1)

    end

  end

  if verbose
      print("\n > INTERIOR POINT SOLVER v0.7 (July 2016)\n\n")
  end

  # ────────────────────────────────────────────────────────────
  #  Initial Point
  # ────────────────────────────────────────────────────────────

  I  = Block([Diag(ones(i)) for i = block_sizes])
  r0 = v4x1(c, d, b, zeros(m,1))
  z  = solve4x4gen(e,I,I)(r0)

  α_v = maxstep(z.v, nothing)
  α_s = maxstep(z.s, nothing)

  # Change to +
  z.v = z.v - α_v*e
  z.s = z.s - α_s*e

  if verbose
      ξ1()=@printf(" %-6s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
                  "  Iter","Mu","prFeas","duFeas","muFeas","refine");ξ1()
  end

  # ────────────────────────────────────────────────────────────
  #  Iterate Loop
  # ────────────────────────────────────────────────────────────

  sol     = Solution(z.y, z.w, z.v, :Error, 0, 0, Inf, Inf, Inf)
  optBest = Inf
  
  for Iter = 1:maxIters

    F    = nt_scaling(z.v, z.s)   # Nesterov-Todd Scaling Matrix
    F⁻¹  = inv(F)
    λ    = F*z.v;                 # This is also F\z.s.

    # The cache is an optimization for the case where there are 
    # many tiny blocks. Since creating mutiple views is time 
    # consuming, we store F and F⁻¹ as big sparse matrices.
    if cache_nestodd == true; cache(F); cache(F⁻¹); end

    solve = solve4x4gen(λ,F,F⁻¹)   # Caches 4x4 solver
                                   # (used a few times, at least 2)

    #         ┌                   ┐ ┌     ┐
    # rleft = │ Q   G'   -A'      │ │ z.y │
    #         │ G                 │ │ z.w │  V = block(F*v)*inv(F)
    #         │ A              -I │ │ z.v │    = block(λ)*λ
    #         │           S     V │ │ z.s │
    #         └                   ┘ └     ┘
    rleft = v4x1( Q*z.y + Gᵀ*z.w - Aᵀ*z.v ,
                  G*z.y                   ,
                  A*z.y - z.s             ,
                  λ ∘ λ                   )

    # True Residual of nonlinear KKT System
    r0 = v4x1(rleft.y - c, rleft.w - d, rleft.v - b, rleft.s);

    # ────────────────────────────────────────────────────────────
    #  Predictor
    # ────────────────────────────────────────────────────────────

    d_aff   = solve(r0)

    α_aff_v = min( maxstep( z.v, d_aff.v ) , 1 )
    α_aff_s = min( maxstep( z.s, d_aff.s ) , 1 )
    α_aff   = min( α_aff_v , α_aff_s )

    μbar = z.v'z.s
    # >> ρ  = (z.v - α_aff*d_aff.v)'*(z.s - α_aff*d_aff.s)/μbar
    ρ  = fts(z.v, α_aff, d_aff.v, z.s, α_aff,d_aff.s)/μbar
    σ  = max(0,min(1,ρ))^3
    μ  = μbar/conedim

    # ────────────────────────────────────────────────────────────
    #  Corrector
    # ────────────────────────────────────────────────────────────

    F⁻¹dfs = F⁻¹*d_aff.s
    Fdfs   = F*d_aff.v

    # >> lc = -(F⁻¹dfs ∘ Fdfs) + (σ*μ)[1]*e;
    lc = (F⁻¹dfs ∘ Fdfs); axpy!(-(σ*μ)[1], e, lc);
    scal!(length(e), -1., lc, 1)

    r  =  v4x1(r0.y, r0.w, r0.v, rleft.s - lc)

    # ────────────────────────────────────────────────────────────
    #  Take newton step, with iterative refinement
    # ────────────────────────────────────────────────────────────

    Δz  = solve(r);
    rStep = 1;
    for rStep = 1:maxRefinementSteps
      r0  = v4x1( Q*Δz.y  + Gᵀ*Δz.w  - Aᵀ*Δz.v ,
                  G*Δz.y                       ,
                  A*Δz.y - Δz.s                ,
                  λ∘(F*Δz.v) + λ∘(F⁻¹*Δz.s) )
      rIr = r - r0
      rnorm = norm(rIr)/(n + 2*m)
      if rnorm > 0.1
        verbose ? warn("4x4 solve failed, residual norm $(rnorm)") : ◂
      end
      if rnorm < refinementThreshold; break; end
      Δzr = solve(rIr)
      Δz  = Δz + Δzr
    end

    α_v = min( maxstep(z.v, Δz.v/(1-DTB)), 1 )
    α_s = min( maxstep(z.s, Δz.s/(1-DTB)), 1 )
    α   = min( α_v, α_s )

    # >> z = z - α*Δz;
    axpy4!(-α, Δz, z)

    # ────────────────────────────────────────────────────────────
    #  Print iterate status
    # ────────────────────────────────────────────────────────────

    rDu = norm(r0.y)/(1+norm(c))
    rPr = normsafe(r0.v)/(1 + normsafe(b))
    rCp = normsafe(r0.s)/(1+abs(c'z.y)[1]); # [1] acts as a cast

    if max(rDu, rPr, rCp) < optBest
      sol.y[:] = z.y; sol.w[:] = z.w; sol.v[:] = z.v
      sol.Iter = Iter; sol.Mu = μ[1]; 
      sol.duFeas = rDu; sol.prFeas = rPr; sol.muFeas = rCp
      optBest = max(rDu, rPr, rCp)
    end

    if verbose
        ξ2()=@printf(" %6i  %-10.4e  %-10.4e  %-10.4e  %-10.4e  %i\n",
                    Iter, μ[1], rDu, rPr, rCp, rStep);ξ2()
    end    

    if max(rDu, rPr, rCp) < optTol
        if verbose
            print("\n > EXIT -- Below Tolerance!\n\n")
        end
        sol.status = :Optimal
        return sol
    end

    # ────────────────────────────────────────────────────────────
    # If iterates blow up, diagonose divergence
    # ────────────────────────────────────────────────────────────
    if max(μ[1], rDu, rPr, rCp) > 1/optTol || !all(isfinite([μ[1], rDu, rPr, rCp]))

        # Primal Infeasible

        r_infeas = norm(Gᵀ*z.w + Aᵀ*z.v)[1]/(b'*z.v + d'*z.w)[1]

        if r_infeas/(1+norm(c)) < optTol

          if verbose;
            print("\n > EXIT -- Infeasible!\n\n")
          end
          sol.status = :Infeasible
          return sol

        end 

        # Dual Infeasible
        
        r_dual_infeas1 = isempty(A) ? -Inf : (A*z.y - z.s)[1]/vecdot(c,z.y)
        r_dual_infeas2 = isempty(G) ? -Inf : norm(G*z.y - d)

        if max(r_dual_infeas1, r_dual_infeas2) < optTol

          if verbose;
            print("\n > EXIT -- Dual Infeasible!\n\n")
          end
          sol.status = :DualInfeasible
          return sol

        end

        # Cause of Divergence Unknown
        
        if verbose
            print("\n > EXIT -- Error!\n\n")
        end
        sol.status = :Error
        return sol

    end

  end

  sol.status = :Abandoned
  return sol

end

end
