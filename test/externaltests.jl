"""
Make sure solver is compatible with external interfaces
AND make sure it returns the right answer.
"""

module MathProgBaseTest

using IntPoint
using MathProgBase

Pkg.installed("MathProgBase")
include(Pkg.dir("MathProgBase")"/test/conicinterface.jl"); 
coniclineartest(IntPointSolver(verbose = false))
conicSDPtest(IntPointSolver(verbose = false, optTol = 1e-6))

end
# Convex.jl Tests

module ConvexTest

installed = true
try
	Pkg.installed("Convex"); 
catch
	installed = false
end

if installed

	using IntPoint
	using Convex; 

	set_default_solver(IntPointSolver(verbose = false, optTol = 1e-6))
	include(Pkg.dir("Convex")"/test/runtests_single_solver.jl")

end

end

# JuMP tests

module JumpTest

installed = true
try
	Pkg.installed("JuMP")
	Pkg.installed("MathProgBase")
	Pkg.installed("FactCheck")
catch
	installed = false
end

if installed

	using JuMP
	using IntPoint
	using FactCheck
	using MathProgBase

	solvers = [IntPointSolver(verbose = false, optTol = 1e-6)]
	sdp_solvers = [IntPointSolver(verbose = false, optTol = 1e-6)]
	conic_solvers_with_duals = [IntPointSolver(verbose = false, optTol = 1e-6)]
	lp_solvers = [IntPointSolver(verbose = false, optTol = 1e-6)]
	ip_solvers = []
	JuUMPdir = Pkg.dir("JuMP")

	grb = false
	cpx = false
	cbc = false
	glp = false
	mos = false

	include(JuUMPdir*"/test/socduals.jl");
	include(JuUMPdir*"/test/probmod.jl");      
	include(JuUMPdir*"/test/sdp.jl");   

end
end