using JuMP, Ipopt

include("modeldata.jl")

type MeanUtilityModel <: AbstractUncertaintyModel
    x::Array{JuMP.Variable}
    λ::JuMP.NonlinearParameter
    model::JuMP.Model
    ysamples::Array{Float64,2}
end

function MeanUtilityModel(data::ModelData, y::MultivariateDistribution;
                          l::Float64 = 1., N::Int = Int(1e4))
    # Create JuMP model to maximise utility: u(x)=(1-exp(-λx))/λ
    # i.e. minimise E[exp(-λx)]
    a = data.a
    B = data.B
    n = length(a)

    srand(0) # Reset seed for reproducibility
    ysamples = rand(y, N)

    m = Model(solver=IpoptSolver(print_level=0))
    x0 = 2mean(y)
    @variable(m, x[i=1:n] >= 0, start=x0[i])

    @NLexpression(m, v[i=1:n], a[i]*exp(-B[i,i]*x[i])*
                  prod(1-exp(-x[j]*B[i,j]) for j=1:n if (j != i) && !iszero(B[i,j])))
    @NLparameter(m, λ == l)
    @NLexpression(m, Eu,
                  sum(exp(-λ*sum((x[i]-ysamples[i,sample])*v[i] for i=1:n)) for
                  sample=1:N)/N)
    @NLobjective(m, Min, Eu)

    # @NLexpression(m, Eu,
    #               sum{1-exp(-λ*sum{(x[i]-ysamples[i,sample])*v[i], i=1:n}),
    #               sample=1:N}/(λ*N))
    # @NLobjective(m, Max, Eu)

    return MeanUtilityModel(x, λ, m, ysamples)
end

#==
modeldata3d, y3d = data3d() # modeldata.jl
srand(0)
modelmeanutil3d = MeanUtilityModel(modeldata3d, y3d, N = Int(1e5))
==#
