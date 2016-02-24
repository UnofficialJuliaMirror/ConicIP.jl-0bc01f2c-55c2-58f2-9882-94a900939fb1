IntPoint.jl: A Pure Julia Conic QP Solver
==

Intpoint is an interior point solver based on [cvxopt](http://cvxopt.org/) for quadratic programs with polyhedral (here denoted `𝑅`) and second order cone (denoted `𝑄`) constraints. Since `Intpoint` is written in Julia, it allows abstract input and allows callbacks for it's most computationaly intensive internal routines.

#### Basic Usage

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

#### Exploiting Structure

The interior point solver can sometimes be sped up dramatically by exploiting the simultanious structure of `Q`,`A` and `G`. For this purpose, we provide a callback function `solve2x2gen`. This is a function of a single argument `F`, which is a `Block` matrix with blocks corrosponding to the cones specified in `𝐾`.
```       
F[i] =  diag(u)    (of type Diag)         if F[i][1] == "R"
F[i] =  αI + uuᵀ   (of type SymWoodbury)  if F[i][1] == "Q"

```
Each block of `F` is positive. `solve2x2gen` is expected to return a function which solves the argumented system.
```
┌                ┐ ┌   ┐   ┌   ┐
│ Q + AᵀF²A   Gᵀ │ │ y │ = │ a │
│ G              │ │ x │   │ b │
└                ┘ └   ┘   └   ┘
```

As an example, the optimization problem

```
minimize    ½yᵀQy - cᵀy
s.t         y ≧ 0
```
has a particularly simple argumented system
```julia

n = 1000

Q = speye(n)
c = ones(n,1)
A = speye(n)
b = zeros(n,1);
𝐾 = [("R",n)]

function solve2x2gen(F)

  # return a function (y,x) ↦ ( (Q + diag(u))⁻¹y , zeros(0,1) )

  HD⁻¹ = cholfact(H + spdiagm( (F[1].diag).^(-2) ))
  return (y, x) -> (HD⁻¹\y, zeros(0,1))

end

@time Sol = intpoint( Q , c , A , b , 𝐾 , G , d , solve2x2gen = solve2x2gen )
```

#### Abstract Matrices
