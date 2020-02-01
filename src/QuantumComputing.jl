module QuantumComputing
include("./QuantumAlgebra.jl")
using .QuantumAlgebra
using PrettyTables
import .QuantumAlgebra: transform!, measure!
using .QuantumAlgebra: normalise!
import Base: show, *, kron, push!, length, ^
import LinearAlgebra: I

export Basis
export Identity

"""
`Qubit` is an wrapper around Ket for simplified Quantum Computing

# Summary
mutable struct Qubit <: Any
# Fields
ψ :: Ket
"""
mutable struct Qubit
    ψ :: Ket
    """
        Qubit(ψ::Array{T,1}, basis::Basis = nothing) where T<:Number

    Make a Qubit in its normalised form in given Basis.

    _Normal Basis considered if no Basis provided_
    """
    function Qubit(ψ::Array{T,1}, basis::Basis ) where T<:Number

        new(Ket(ψ, basis))
    end
    function Qubit(ψ::Array{T,1}) where T<:Number
        new(Ket(ψ))
    end
end # Qubit
export Qubit

mutable struct Operator
    symbol :: String
    matrix :: Array{Number, 2}
    number_of_bits :: Integer
    """
        Operator(matrix :: Array{T,2}, basis :: Basis)
    Creates an Operator.

        `matrix` is a n ⨯ n matrix in provided basis.

        _If no basis is provided, Identity basis will be considered._
    """
    function Operator(symbol::String, matrix::Array{Any, 2})
        Operator(symbol, convert(Array{Number,2}, matrix))
    end
    function Operator(symbol::String, matrix::Array{T,2}, basis :: Basis) where T <: Number
        length(matrix) % 4 == 0 && round.(adjoint(matrix)*matrix; digits = 6) == I  || throw("NotUnitaryError: matrix not unitary")
        inv(basis.transformMatrix)*matrix*inv(basis.transformMatrix)
        new(symbol, matrix, size(matrix,1)/2)
    end
    function Operator(symbol::String, matrix::Array{T,2}) where T <: Number
        length(matrix) % 4 == 0 && round.(adjoint(matrix)*matrix; digits = 6) == I  || throw("NotUnitaryError: matrix not unitary")
        new(symbol, matrix, size(matrix,1)/2)
    end
end # Operator
export Operator

struct Circuit
    circuit :: Array{Operator,1}
    bits_to_operate :: Array{Array{Int64,1},1}
    input_size :: Int64
    """
        Circuit()
    Creates an empty Circuit. Use `push!(circuit, operator, bits_to_operate)` to populate
    """
    function Circuit(input_size::Int64)
        new([], [], input_size)
    end # function
end # struct
export Circuit

function Base.length(circuit::Circuit) :: Int64
    return length(circuit.circuit)
end

"""
    Base.push!(circuit::Circuit, operator::Operator, bits_to_operate::Array{Int64,1})

Overloaded from Base

Add an operator into the circuit

    `bits_to_operate` is an array that holds the locations of the bits that the operator is to operate on.
"""
function Base.push!(circuit::Circuit, operator::Operator, bits_to_operate::Array{Int64,1}) :: Circuit
    if operator.number_of_bits != length(bits_to_operate)
        throw("IncorrectInputNumberError: Expected ", operator.number_of_bits, ", got ", length(bits_to_operate))
    elseif any(x->x>circuit.input_size, bits_to_operate)
        throw("IncorrectOperatingQubitError: Circuit size =", circuit.input_size,", got qubits ", filter( x-> x>circuit.input_size, bits_to_operate))
    elseif !allunique(bits_to_operate)
        throw("FanOutError: Operator cannot operate on the same qubit as two inputs.")
    else
        push!(circuit.circuit, operator)
        push!(circuit.bits_to_operate, bits_to_operate)
        return circuit
    end
end
export push!

"""
    u_push!(circuit::Circuit, operator::Operator, bits_to_operate::Array{Int64,1})

Add an operator into the circuit without checking for errors

*Use `push!` instead for small operations*

`bits_to_operate` is an array that holds the locations of the bits that the operator is to operate on.
"""
function u_push!(circuit::Circuit, operator::Operator, bits_to_operate::Array{Int64,1}) :: Circuit
    push!(circuit.circuit, operator)
    push!(circuit.bits_to_operate, bits_to_operate)
    return circuit
end
export u_push!

"""
    QuantumAlgebra.transform(operator::Operator, newbasis::Basis) :: Operator

Overloaded from QuantumAlgebra

Transform an operator from its Basis to another and return the new operator.
"""
function transform(operator::Operator, newbasis::Basis) :: Operator
    Operator(operator.symbol, newbasis.transformMatrix * operator.matrix * newbasis.transformMatrix)
end
export transform
"""
    Base.show(io::IO, ψ::Qubit)

Overloaded from Base
Pretty print Qubit
"""
function Base.show(io::IO, q::Qubit)
    print(io,q.ψ)
end

"""
    Base.show(io::IO, circuit::Circuit)

Overloaded from Base
Pretty print Circuit
"""
function Base.show(io::IO, circuit::Circuit)
    print(
        join(
            [
                circuit.circuit[i].symbol *
                "(" *
                join(
                    circuit.bits_to_operate[i],
                    ", "
                )*
                ")"
                for i in 1:length(circuit) ],
            " --> "
        )
    )
end

"""
    pretty_print_operator_matrix(operator::Operator)

An unexported helper to pretty print the operator matrix
"""
function pretty_print_operator_matrix(io::IO, operator::Operator)
    pretty_table(io, operator.matrix, alignment=:c, noheader=true, screen_size=(-1,-1), formatter = Dict(0 => (v,i) -> round(v,digits=4)), tf=unicode_matrix)
end # function

"""
    Base.show(io::IO, operator::Operator)

Overloaded from Base
Pretty print Operator
"""
function Base.show(io::IO, operator::Operator)
    println(operator.symbol, "(", operator.number_of_bits,")")
    pretty_print_operator_matrix(io, operator)
end

export show

"""
    operate!(operator::Operator, qubit::Qubit)

Operates an operator on a qubit
"""
function operate!(operator::Operator, qubit::Qubit) :: Qubit
    if operator.number_of_bits == length(qubit.ψ.coefficients)/2
        qubit.ψ.coefficients = transform(operator, qubit.ψ.basis).matrix * qubit.ψ.coefficients
        normalise!(qubit.ψ)
        return qubit
    else
        throw("IncorrectInputNumberError: Expected ", operator.number_of_bits, ", got ", Int(length(qubit.ψ.coefficients)/2))
    end
    return qubit
end # function operate!
export operate!
function operate!(operator::Operator, qubits::Array{Qubit, 1}) :: Array{Qubit, 1}
    if operator.number_of_bits == length(qubits)
        operate!(operator, kron(qubits))
    else
        throw("IncorrectInputNumberError: Expected ", operator.number_of_bits, ", got ", length(qubits))
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
        return Operator(operator1.symbol*"*"*operator2.symbol, operator1.matrix*operator2.matrix)
    end
    throw("InputNumberMismatch: Operators have different number of inputs")
    return operator1
end
export *

"""
    Base.^(op::Operator, n::Int64)

Returns the n<sup>th</sup> tensor product of the operator
"""
function ^(op::Operator, n::Int64) :: Operator
    return Operator(op.symbol*"^"*string(n), kron(fill(op, n)))
end # function


"""
    kron(q1::Qubit, q2::Qubit)

kron overload for Qubits
"""
function Base.kron(q1::Qubit, q2::Qubit) :: Qubit
    return Qubit(Array{Number}(kron(q1.ψ.coefficients, q2.ψ.coefficients)))
end # function
"""
    kron(qs::Array{Qubit})

kron overload for Qubits
"""
function Base.kron(qs::Array{Qubit, 1}) :: Qubit
    return foldr(kron,qs)
end # function

"""
    kron(op1::Operator, op2::Operator)

kron overload for Operators
"""
function kron(op1::Operator, op2::Operator) :: Operator
    return Operator(op1.symbol*"⊗"*op2.symbol, Array{Number}(kron(op1.matrix, op2.matrix)))
end # function

"""
    kron(ops::Array{Operator})

kron overload for Operators
"""
function Base.kron(ops::Array{Operator, 1}) :: Operator
    return foldr(kron,ops)
end # function

⊗ = Base.kron
export kron, ⊗
"""
    measure(qubit::Qubit) :: Int64

Measure the qubit and destroy the qubit
"""
function QuantumAlgebra.measure!(qubit::Qubit) :: Qubit
    qubit.ψ = measure!(qubit.ψ)
    return qubit
end # function
M = measure!
export measure!, M

function make_controled(operator::Operator) :: Operator
    return Operator()
end

const Identity = Operator("I", [1 0; 0 1])
const Hadamard = Operator("H", 1/√2 * [1 1; 1 -1])
const H = Hadamard
const PauliX = Operator("NOT", [0 1; 1 0])
const NOT = PauliX
const PauliY = Operator("P-Y", [0 -1im; 1im 0])
const Z = Operator("P-Z", [1 0; 0 -1])
const PauliZ = Z
export Hadamard, H, PauliX, NOT, PauliY, PauliZ, Z, Identity
const comp_basis = Basis(1/√2 * [1 1; 1 -1], "Computational Basis")
export comp_basis
end #QuantumComputing
