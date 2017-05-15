# ──────────────────────────────────────────────────────────────
#
# Diag
# Wrapper around Diagonal, extending its functionality.
# NOTE: we do not extend Diagonal to avoid contaminating the base
# NOTE: Julia 0.5 has all the functionality needed here. Can
# 		replace with the generic Diagonal once it comes out.
#       I'm looking forward to getting rid of this forever.
#
# ──────────────────────────────────────────────────────────────

import Base:+,*,-,\,^,sparse

ViewTypes   = Union{SubArray}
VectorTypes = Union{AbstractMatrix, Vector, ViewTypes}
MatrixTypes = Union{AbstractMatrix, Array{Real,2},
                    SparseMatrixCSC{Real,Integer}}

type Diag
    diag::Vector
end

+(A::Diag,B::Diag)             = Diag(A.diag + B.diag)
+(A::Diag,T::UniformScaling)   = Diag(A.diag + T.λ)
+(T::UniformScaling,A::Diag)   = Diag(A.diag + T.λ)
*(T::UniformScaling,A::Diag)   = Diag(A.diag*T.λ)
*(A::Diag,T::UniformScaling)   = Diag(A.diag*T.λ)
*(α::Real,A::Diag)             = Diag(A.diag*α)
*(A::Diag,α::Real)             = Diag(A.diag*α)
*(A::Diag,B::Diag)             = Diag(A.diag.*B.diag)
function +(A::MatrixTypes,B::Diag)
    O = copy(A)
    for i = 1:size(A,2); O[i,i] = A[i,i] + B.diag[i]; end
    return O
end
*(A::Diag,B::AbstractMatrix)   = A.diag.*B
*(A::Diag,b::AbstractVector)   = A.diag.*b
+(B::Diag,A::MatrixTypes)      = A + B
^(A::Diag, n::Integer)         = Diag(A.diag.^n)
Base.full(A::Diag)             = full(Diagonal(A.diag))
Base.inv(A::Diag)              = Diag(1.0./A.diag)
*(A::Diag, b::ViewTypes)       = A.diag.*b
Base.size(A::Diag, k::Integer) = (k == 1 || k == 2) ? size(A.diag,1) : 1
Base.size(A::Diag)             = (size(A.diag,1), size(A.diag,1))
Id(n::Integer)                 = Diag(ones(n))
Base.sparse(A::Diag)           = spdiagm(A.diag)

Base.Ac_mul_B(O1::Diag, O2::Diag) = O1*O2

# returns a pointer to the original matrix, this is consistent with the
# behavior of Symmetric in Base.
Base.ctranspose(A::Diag) = A 
Base.Ac_mul_B(A::Diag, x::VectorTypes) = A*x
