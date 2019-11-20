module QuantumComputing
include("./QuantumAlgebra.jl")
using .QuantumAlgebra
import .QuantumAlgebra: transform!, measure!
import Base: show, *
import LinearAlgebra: I

export Basis
export Identity
mutable struct Qubit
    ψ :: Ket
    """
        Qubit(ψ::Array{T,1}, basis::Basis = nothing) where T<:Number

    Make a Qubit in its normalised form in given Basis.
    Normal Basis considered if no Basis provided
    """
    function Qubit(ψ::Array{T,1}, basis::Basis ) where T<:Number
        if length(ψ) == 2
            new(Ket(ψ, basis))
        else
            throw("IncorrectSizeError: Qubits must be in a 2 dimension basis")
        end
    end
    function Qubit(ψ::Array{T,1}) where T<:Number
        if length(ψ) == 2
            new(Ket(ψ))
        else
            throw("IncorrectSizeError: Qubits must be in a 2 dimension basis")
        end
    end
end # Qubit
export Qubit

mutable struct Operator
    matrix :: Array{Number, 2}
    number_of_bits :: Integer
    """
        Operator(matrix :: Array{T,2}, basis :: Basis)
    Creates an Operator.
        matrix is a n ✖ n matrix in provided basis.
        If no basis is provided, normal basis will be considered.
    """
    function Operator(matrix :: Array{Any, 2})
        Operator(convert(Array{Number,2}, matrix))
    end
    function Operator(matrix :: Array{T,2}, basis :: Basis) where T <: Number
        length(matrix) % 4 == 0 && round.(adjoint(matrix)*matrix; digits = 6) == I  || throw("NotUnitaryError: matrix not unitary")
        inv(basis.transformMatrix)*matrix*inv(basis.transformMatrix)
        new(matrix, size(matrix,1)/2)
    end
    function Operator(matrix :: Array{T,2}) where T <: Number
        length(matrix) % 4 == 0 && round.(adjoint(matrix)*matrix; digits = 6) == I  || throw("NotUnitaryError: matrix not unitary")
        new(matrix, size(matrix,1)/2)
    end
end # Operator
export Operator

"""
    QuantumAlgebra.transform(operator::Operator, newbasis::Basis) :: Operator

Overloaded from QuantumAlgebra
Transform an operator from its Basis to another and return the new operator.
"""
function transform(operator::Operator, newbasis::Basis) :: Operator
    Operator(newbasis.transformMatrix * operator.matrix * newbasis.transformMatrix)
end

"""
    Base.show(io::IO, ψ::Qubit)

Overloaded from Base
Pretty print Qubit
"""
function Base.show(io::IO, q::Qubit)
    #println(io, "Hello")
    print(io,q.ψ)
end
export show
"""
    operate!(operator::Operator, qubit::Qubit)

Operates an operator on a qubit
"""
function operate!(operator::Operator, qubit::Qubit) :: Qubit
    if operator.number_of_bits == 1
        qubit.ψ.coefficients = transform(operator, qubit.ψ.basis).matrix * qubit.ψ.coefficients
    else
        throw("IncorrectInputNumberError: Expected $(operator.number_of_bits), got 1")
    end
    return qubit
end # function operate!
export operate!
function operate!(operator::Operator, qubits::Array{Qubit, 1}) :: Array{Qubit, 1}
    if operator.number_of_bits == length(qubits)
        throw("UnderConstructionError:")
    else
        throw("IncorrectInputNumberError: Expected $(operator.number_of_bits), got $(length(qubits))")
    end
    return qubits
end # function

"""
    Base.*(operator::Operator, qubit::Qubit)

Alias for operate!(). Changes the qubit
"""
function *(operator::Operator, qubit::Union{Qubit, Array{Qubit, 1}}) :: Qubit
    return operate!(operator, qubit)
end # function *
export *

function *(operator1::Operator, operator2::Operator) :: Operator
    if operator1.number_of_bits == operator2.number_of_bits
        return Operator(operator1.matrix*operator2.matrix)
    end
    throw("InputNumberMismatch: Operators have different number of inputs")
    return operator1
end
export *

"""
    measure(qubit::Qubit) :: Int64

Measure the qubit and delete it
"""
function QuantumAlgebra.measure!(qubit::Qubit)
    qubit.ψ = measure!(qubit.ψ)
    return qubit
end # function
M = measure!
export measure!, M

const Hadamard = Operator(1/√2 * [1 1; 1 -1])
const H = Hadamard
const PauliX = Operator([0 1; 1 0])
const NOT = PauliX
const PauliY = Operator([0 -1im; 1im 0])
const Z = Operator([1 0; 0 -1])
const PauliZ = Z
export Hadamard, H, PauliX, NOT, PauliY, PauliZ, Z
const comp_basis = Basis(1/√2 * [1 1; 1 -1], "Computational Basis")
export comp_basis
end #QuantumComputing
