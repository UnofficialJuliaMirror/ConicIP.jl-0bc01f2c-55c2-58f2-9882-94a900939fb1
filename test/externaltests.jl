"""
Make sure solver is compatible with external interfaces
AND make sure it returns the right answer.
"""

module MathProgBaseTest

using ConicIP
using MathProgBase

Pkg.installed("MathProgBase")
include(Pkg.dir("MathProgBase")"/test/conicinterface.jl"); 
coniclineartest(ConicIPSolver(verbose = false))
conicSDPtest(ConicIPSolver(verbose = false, optTol = 1e-7))

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

	using ConicIP
	using Convex; 

	set_default_solver(ConicIPSolver(verbose = false, optTol = 1e-7))
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
	using ConicIP
	using FactCheck
	using MathProgBase

	solvers = [ConicIPSolver(verbose = false, optTol = 1e-6)]
	sdp_solvers = [ConicIPSolver(verbose = false, optTol = 1e-6)]
	conic_solvers_with_duals = [ConicIPSolver(verbose = false, optTol = 1e-6)]
	lp_solvers = [ConicIPSolver(verbose = false, optTol = 1e-6)]
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