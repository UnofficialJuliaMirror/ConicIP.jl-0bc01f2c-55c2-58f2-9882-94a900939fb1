using IntPoint
using ECOS
using Convex
#using ProfileView
using PyPlot
using MUMPS

icntl = default_icntl[:];
icntl[1] = 0;
icntl[2] = 0;
icntl[3] = 0;
icntl[4] = 0;

function mykktsolver(Q, A, G)

  mumps = Mumps{Float64}(mumps_symmetric, icntl, default_cntl64);

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  Q = sparse(Q) # TODO: remove the need for this ?
  A = sparse(A)
  G = sparse(G)

  function solve3x3gen_sparse(F, F⁻ᵀ)

	  (FᵀFA, FᵀFB, invFᵀFD) = IntPoint.lift(F'F);r  = size(invFᵀFD,1)

	  Z = [ Q             G'             A'            spzeros(n,r)
	        G             spzeros(p,p)   spzeros(p,m)  spzeros(p,r)
	        A             spzeros(m,p)  -FᵀFA          FᵀFB                         
	        spzeros(r,n)  spzeros(r,p)   FᵀFB'        -invFᵀFD        ]

	  factorize(mumps, Z)

	  function solve3x3lift(Δy, Δw, Δv)
	    z = solve(mumps,[Δy; Δw; Δv; zeros(r,1)][:])
	    return (z[1:n,:], z[n+1:n+p,:], -z[(n+p+1):(n+m+p),:])
	  end

	 return solve3x3lift

  end

  return solve3x3gen_sparse

end

function svm(X, solver=IntPointSolver(); C = 1.0)
    n = size(X,2)
    w = Variable(n)
    obj = C*sumsquares(w) + sum(max(1-X*w, 0))
    problem = minimize(obj)
    solve!(problem, solver)
    return evaluate(w)
end

X = rand(200,200)

IntSolver = IntPointSolver(verbose = false, 
						   kktsolver = mykktsolver,
						   maxRefinementSteps = 0,
						   preprocess = false)

Profile.clear()
@time x1 = svm(X, IntSolver)

ESolver = ECOSSolver(verbose = false, reltol = 1e-6)
@time x2 = svm(X, ESolver)

norm(x1 - x2, Inf)
