# ──────────────────────────────────────────────────────────────
#  Various KKT Solvers
# ──────────────────────────────────────────────────────────────

"""
Solves the 3x3 system
```
┌             ┐ ┌    ┐   ┌   ┐
│ Q   G'  -A' │ │ y' │ = │ y │
│ G           │ │ w' │   │ w │ 
│ A       FᵀF │ │ v' │   │ v │
└             ┘ └    ┘   └   ┘
```
by the double QR method described in CVXOPT
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
section 10.2
"""
function kktsolver_qr(Q, A, G)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  (Q0,R1) = qr(full(G'), thin = false)
  Q1      = Q0[:,1:p]
  Q2      = Q0[:,p+1:end]

  function solve3x3gen(F, F⁻ᵀ)

    F⁻ᵀ     = full(inv(F))'
    Atil    = F⁻ᵀ*full(A)
    L = qrfact(Q2'*(Q + Atil'Atil)*Q2)

    function solve3x3(bx, by, bz)

      Q1ᵀx = R1'\by
      Q2ᵀx = L \ ( Q2'*(bx + Atil'*(F⁻ᵀ*bz)) -
                   Q2'*((Q + Atil'Atil)*(Q1*(Q1ᵀx))) )
      y    = R1 \ ( Q1'*(bx + Atil'*(F⁻ᵀ*bz))    -
                    Q1'*(Q + Atil'Atil)*Q1*Q1ᵀx -
                    Q1'*(Q + Atil'Atil)*Q2*Q2ᵀx )
      x    = Q0'\[Q1ᵀx; Q2ᵀx]
      Fz   = ( F⁻ᵀ*bz - 
               Atil*(Q1*(Q1ᵀx)) - Atil*(Q2*(Q2ᵀx)) )
      z    = inv(F)*Fz

      return (x,y,z)

    end

    return solve3x3

  end

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

"""
Estimates for the number of nonzeros of lift(F)
"""
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

"""
Estimates the number of nonzeros of F
"""
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

""" 
Solves the 3x3 system
```
┌             ┐ ┌    ┐   ┌   ┐
│ Q   G'  -A' │ │ y' │ = │ y │
│ G           │ │ w' │   │ w │ 
│ A       FᵀF │ │ v' │   │ v │
└             ┘ └    ┘   └   ┘
```

By lifting the large diagonal plus rank 3 blocks of FᵀF

Intelligently chooses between solve3x3gen_sparse_lift and
solve3x3gen_sparse_dense by approximating the number of non-zeros in
both and choosing the form with more sparsity. The former is better
for large second order cones, while the latter is better if the
constraints are the product of many small cones. 
"""
function kktsolver_sparse(Q, A, G)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  Q = sparse(Q) # TODO: remove the need for this ?
  A = sparse(A)
  G = sparse(G)

  function solve3x3gen_sparse(F, F⁻ᵀ)

    if count_lift(F) < count_dense(F)
    
      (FᵀFA, FᵀFB, invFᵀFD) = lift(F'F);r  = size(invFᵀFD,1)

      Z = [ Q             G'            -A'            spzeros(n,r)
            G             spzeros(p,p)   spzeros(p,m)  spzeros(p,r)
            A             spzeros(m,p)   FᵀFA          FᵀFB                         
            spzeros(r,n)  spzeros(r,p)   FᵀFB'         invFᵀFD        ]

      ZZ = lufact(Z)

      function solve3x3lift(Δy, Δw, Δv)
        z = ZZ\[Δy; Δw; Δv; zeros(r,1)]
        return (z[1:n,:], z[n+1:n+p,:], z[(n+p+1):(n+m+p),:])
      end

     return solve3x3lift

    else

      FᵀF = sparse(F'F)

      Z = [ Q        G'             -A' 
            G        spzeros(p,p)   spzeros(p,m)
            A        spzeros(m,p)   FᵀF           ]

      ZZ = lufact(Z)

      function solve3x3dense(Δy, Δw, Δv)
        z = ZZ\[Δy; Δw; Δv]
        return (z[1:n,:], z[n+1:n+p,:], z[n+p+1:end,:])
      end

      return solve3x3dense

    end    


  end

  return solve3x3gen_sparse

end
"""
Solves the 2x2 system
```
┌                   ┐ ┌    ┐   ┌   ┐
│ Q + A'F⁻¹F⁻ᵀA  G' │ │ y' │ = │ y │
│ G                 │ │ w' │   │ w │ 
└                   ┘ └    ┘   └   ┘
```
"""
function kktsolver_2x2(Q, A, G)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  function solve2x2gen(F, F⁻ᵀ)

    F⁻ᵀ = sparse(F⁻ᵀ)
    AᵀF⁻¹F⁻ᵀA = A'*(F⁻ᵀ'*(F⁻ᵀ*A))

    Z = [ Q + AᵀF⁻¹F⁻ᵀA   G'            
          G            spzeros(p,p) ]

    Z = lufact(Z)

    function solve2x2(Δy, Δw)

      z = Z\[Δy; Δw]
      return (z[1:n,:], z[n+1:end,:])

    end

    return solve2x2

  end

  return solve2x2gen

end

"""
Wrapper around solve2xegen to solve 3x3 systems by pivoting
on the third component.
"""
function pivotgen(kktsolver_2x2,Q,A,G)

  solve2x2gen = kktsolver_2x2(Q,A,G)

  function solve3x3gen(F, F⁻ᵀ)

    solve2x2 = solve2x2gen(F, F⁻ᵀ)

    function solve3x3(y, w, v)

      t1 = F⁻ᵀ*(F⁻ᵀ*v)
      (Δy, Δw) = solve2x2(y + A'*t1, w)
      axpy!(-1, F⁻ᵀ*(F⁻ᵀ*(A*Δy)), t1)  # Δv = F⁻²*(v - A*Δy)

      return(Δy, Δw, t1)

    end

  end

  return solve3x3gen

end

pivot(kktsolver_2x2) = (Q,A,G) -> pivotgen(kktsolver_2x2,Q,A,G)
