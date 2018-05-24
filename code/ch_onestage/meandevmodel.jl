using JuMP, Ipopt

include("modeldata.jl")

type MeanDeviationModel <: AbstractUncertaintyModel
    x::Array{JuMP.Variable}
    λ::JuMP.NonlinearParameter
    model::JuMP.Model
    # TODO: make μy and Σy into NLparameters too?
end

function MeanDeviationModel(data::ModelData, μ::Array{Float64},
                            Σ::Array{Float64,2}; l::Float64 = 1.)
    # Create JuMP model with objective:  expected value - λ*standard deviation
    a = data.a
    B = data.B
    n = length(a)

    m = Model(solver=IpoptSolver(print_level=0))
    x0 = 2μ
    @variable(m, x[i=1:n] >= 0, start=x0[i])
    @NLparameter(m, λ == l)

    @NLexpressions m begin
        demand[i=1:n], (a[i]*exp(-B[i,i]*x[i])*
                        prod(1-exp(-B[i,j]*x[j]/x[i]) for j=1:n
                             if (j != i) && !iszero(B[i,j])))
        Ef[i=1:n], (x[i]-μ[i])*demand[i]
        Varf, sum(Σ[i,j]*demand[i]*demand[j] for j=1:n for i=1:n if Σ[i,j] != 0)
        meandev, sum(Ef[i] for i=1:n) - λ*sqrt(Varf)
    end

    @NLobjective(m, Max, meandev)

    MeanDeviationModel(x,λ,m)
end
