ConicIP.jl: A Pure Julia Conic QP Solver
==
![Test Status](https://travis-ci.org/MPF-Optimization-Laboratory/ConicIP.jl.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/MPF-Optimization-Laboratory/ConicIP.jl/badge.svg?branch=master)](https://coveralls.io/github/MPF-Optimization-Laboratory/ConicIP.jl?branch=master)

`ConicIP` (IP stands for Interior Point, *not* Integer Programming) is an interior point solver inspired by [cvxopt](http://cvxopt.org/) for quadratic programs with polyhedral (here denoted `𝑅`) and second order cone (denoted `𝑄`) constraints. Since `ConicIP` is written in Julia, it allows abstract input and allows callbacks for it's most computationaly intensive internal routines.

#### Basic Usage

ConicIP has the interface
```julia
Sol = conicIP( Q , c , A , b , 𝐾 , G , d )
```
For the problem
```
minimize    ½yᵀQy - cᵀy
s.t         Ay ≧𝐾 b,  𝐾 = 𝐾₁  × ⋯ × 𝐾ⱼ
            Gy  = d
```

`𝐾` is a list of tuples of the form `(Cone Type ∈ {"R", "Q"}, Cone Dimension)` specifying the cone `𝐾ᵢ`. For example, the cone `𝐾 = 𝑅² × 𝑄³ × 𝑅²` has `𝐾`

```julia
𝐾 = [ ("R",2) , ("Q",3),  ("R",2) ]
```

ConicIP returns `Sol`, a structure containing error information (`Sol.status`), the primal variables (`Sol.y`), dual variables (`Sol.v`, `Sol.w`), and convergence information.

To solve the problem

```
minimize    ½yᵀQy - cᵀy
s.t.        y ≧ 0
```

for example, use `ConicIP` as follows

```julia
using ConicIP

n = 1000

Q = sparse(randn(n,n));
Q = Q'*Q;
c = ones(n,1);
A = speye(n);
b = zeros(n,1);
𝐾 = [("R",n)];

sol = conicIP( Q , c , A , b , 𝐾 , verbose = true);
```

For a more detailed example involving callback functions, refer to this
[notebook](https://cdn.rawgit.com/MPF-Optimization-Laboratory/ConicIP.jl/master/examples/callback.html).

### Usage with modelling libraries

ConicIP is integrated with [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) and can be used as a solver in [JuMP](https://github.com/JuliaOpt/JuMP.jl) and [Convex](https://github.com/JuliaOpt/Convex.jl).

#### JuMP.jl

```julia
using JuMP
using ConicIP

m = Model(solver = ConicIPSolver())
@variable(m, x[1:10] >= 0)
@constraint(m, sum(x) == 1.0)
@objective(m, Min, sum(x))
status = solve(m)
getvalue(x) # should be [0.1 0.1 ⋯ 0.1]
```

#### Convex.jl

```julia
using Convex
using ConicIP

set_default_solver(ConicIPSolver())
x = Variable(10)
p = minimize( sum(x), [x >= 0, sum(x) == 1])
solve!(p)
x # should be [0.1 0.1 ⋯ 0.1]
```
