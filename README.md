IntPoint.jl: A Pure Julia Conic QP Solver
==

Intpoint is an interior point solver based on [cvxopt](http://cvxopt.org/) for quadratic programs with polyhedral (here denoted `𝑅`) and second order cone (denoted `𝑄`) constraints. Since `Intpoint` is written in Julia, it allows abstract input and allows callbacks for it's most computationaly intensive internal routines.

#### Usage

Intpoint has the interface
```julia
Sol = intpoint( Q , c , A , b , 𝐾 , G , d )
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

Intpoint returns `Sol`, a structure containing error information (`Sol.status`), the primal variables (`Sol.y`), dual variables (`Sol.v`, `Sol.w`), and convergence information.

To solve the problem

```
minimize    ½yᵀQy - cᵀy
s.t.        y ≧ 0
```

for example, use `IntPoint` as follows

```
using IntPoint

n = 1000

Q = sparse(randn(n,n));
Q = Q'*Q;
c = ones(n,1);
A = speye(n);
b = zeros(n,1);
𝐾 = [("R",n)];

sol = intpoint( Q , c , A , b , 𝐾 , verbose = true);
```

For a more detailed example involving callback functions, refer to this
notebook.
