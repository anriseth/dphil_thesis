using JuMP, Ipopt

include("modeldata.jl")

type CvarModel <: AbstractUncertaintyModel
    x::Array{JuMP.Variable}
    λ::JuMP.NonlinearParameter
    model::JuMP.Model
    ysamples::Array{Float64,2}
    α::JuMP.NonlinearParameter
end

function CvarModel(data::ModelData, y::MultivariateDistribution;
                   l::Float64 = 1., N::Int = Int(1e4),
                   smoothing::Float64 = 2e3)
    # TODO: create a continuation method on the smoothing parameter

    # Create JuMP model to expected shortfall / cvar etc.
    a = data.a
    B = data.B
    n = length(a)

    srand(0) # Reset seed for reproducibility
    ysamples = rand(y, N)

    m = Model(solver=IpoptSolver(print_level=0))
    x0 = 2mean(y)
    @variable(m, x[i=1:n] >= 0, start=x0[i])
    @variable(m, η, start = sum(x0))
    @NLexpression(m, v[i=1:n], a[i]*exp(-B[i,i]*x[i])*
                  prod(1-exp(-B[i,j]*x[j]/x[i]) for j=1:n if (j != i) && !iszero(B[i,j])))
    @NLparameter(m, λ == l)
    @NLparameter(m, α == smoothing)

    @NLexpression(m, f[sample=1:N], sum((x[i]-ysamples[i,sample])*v[i] for i=1:n))
    @NLexpression(m, Esmooth,
                  sum((η-f[sample])+1/α*log(1+exp(-α*(η-f[sample])))
                      for sample=1:N)/N)

    @NLobjective(m, Max, η - (1/λ)*Esmooth)

    return CvarModel(x, λ, m, ysamples, α)
end

#==
modeldata3d, y3d = data3d() # modeldata.jl
modelcvar3d = CvarModel(modeldata3d, y3d, N = Int(1e5))
==#
