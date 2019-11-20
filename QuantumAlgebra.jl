module QuantumAlgebra
using LinearAlgebra
using StatsBase
import Base: *, ==, show

struct Basis
    transformMatrix::Array{Number, 2}
    name::String
    function Basis(transformMatrix::Array{T, 2}, name::String) where T<:Number
        length(transformMatrix) % 4 == 0 && round.(adjoint(transformMatrix)*transformMatrix; digits = 6) == I  || throw("NotUnitaryError: transformMatrix not unitary")
            new(transformMatrix, name)
    end
end
export Basis
function Base.show(io::IO, b::Basis)
    println("Basis: $(b.name)")
    show(io, "text/plain", b.transformMatrix)
end
function Identity(n::Integer = 2) :: Basis
    return Basis(Int.(Array(I, n, n)),"Normal")
end
export Identity
mutable struct Bra
    coefficients::Array{Number, 2}
    basis::Basis
    function Bra(coefficients::Array{T, 2}, basis::Basis) where T<:Number
        if size(coefficients, 1)==1||throw("RowError: Not a row vector") && size(coefficients, 2)%2 == 0 || throw("SizeError: Size must be 2n") && length(coefficients)==size(basis.transformMatrix, 1) || throw("IncorrectBase: This vector cannot be in this basis")
            new(coefficients/√sum(coefficients.^2), basis)
        end
    end
    function Bra(coefficients::Array{T, 2}) where T<:Number
        if size(coefficients, 1)==1||throw("RowError: Not a row vector") && size(coefficients, 2)%2 == 0 || throw("SizeError: Size must be 2n")
            new(coefficients/√sum(coefficients.^2), Identity(length(coefficients)))
        end
    end
end
export Bra
function Base.show(io::IO, ψ::Bra)
    print("⟨$(join(ψ.coefficients,','))|")
end

mutable struct Ket
    coefficients::Array{Number, 1}
    basis::Basis
    function Ket(coefficients::Array{T, 1}, basis::Basis) where T<:Number
        if length(coefficients)%2 == 0 || throw("IncorrectSizeError: Size must be 2n") && length(coefficients)==size(basis.transformMatrix, 1) || throw("IncorrectBase: This vector cannot be in this basis")
            new(coefficients/√sum(coefficients.^2),basis)
        end
    end
    function Ket(coefficients::Array{T, 1}) where T<:Number
        if length(coefficients)%2 == 0 || throw("IncorrectSizeError: Size must be 2n")
            new(coefficients/√sum(coefficients.^2), Identity(length(coefficients)))
        end
    end
end
export Ket
function Base.show(io::IO, ψ::Ket)
    print(io,"|$(join(ψ.coefficients,", "))⟩")
end
export show

# All Functions defined here
function transform!(ψ::Bra, newbasis::Basis)
    if ψ.basis!=newbasis
        ψ.coefficients = ψ.coefficients * inv(ψ.basis.transformMatrix) * newbasis.transformMatrix;
        ψ.basis = newbasis;
    end
    return ψ
end
function transform!(ψ::Ket, newbasis::Basis)
    if ψ.basis!=newbasis
        ψ.coefficients = newbasis.transformMatrix * inv(ψ.basis.transformMatrix) * ψ.coefficients;
        ψ.basis = newbasis;
    end
    return ψ
end
export transform!

function measure!(ket::Ket) :: Ket
    measured = sample(1:length(ket.coefficients), Weights(abs2.(ket.coefficients)))
    ket.coefficients = zeros(length(ket.coefficients))
    ket.coefficients[measured] = 1
    return ket
end
function dual(ψ::Bra) :: Ket
    return Ket(reshape(collect(Number,ψ.coefficients'), length(ψ.coefficients)), ψ.basis)
end
function dual(ψ::Ket) :: Bra
    return Bra(collect(Number, ψ.coefficients'), ψ.basis)
end
export dual

function scalarProduct(ψ1::Bra, ψ2::Ket) :: Complex
    ψ1.basis == ψ2.basis || throw("basisMismatch: Bra and Ket are in different basis")
    return Complex((ψ1.coefficients*ψ2.coefficients)[1])
end
function scalarProduct(ψ1::Ket, ψ2::Ket) :: Complex
    ψ1.basis == ψ2.basis || throw("basisMismatch: Two kets are in different basis")
    return Complex((dual(ψ1).coefficients*ψ2.coefficients)[1])
end
export scalarProduct

function *(ψ1::Union{Ket,Bra}, ψ2::Ket) :: Complex
    return scalarProduct(ψ1,ψ2)
end
function *(b::Basis, ψ::Ket)
    return transform!(b,ψ)
end
function *(ψ::Bra, b::Basis)
    return transform!(ψ,b)
end
export *

function ==(b1::Basis, b2::Basis) :: Bool
    return b1.transformMatrix == b2.transformMatrix
end
function ==(psi1::Ket, psi2::Ket) :: Bool
    return ps1.coefficients==ps2.coefficients && ps1.basis == ps2.basis
end
function ==(psi1::Bra, psi2::Bra) :: Bool
    return ps1.coefficients==ps2.coefficients && ps1.basis == ps2.basis
end
export ==

function norm(ψ::Union{Ket,Bra}) :: Real
    return √sum(ψ.coefficients.^2)
end
export norm

function normalise!(ψ::Union{Ket,Bra})
    ψ.coefficients = ψ.coefficients/norm(ψ)
    return ψ
end
export normalise!

end  # module QuantumComputing
