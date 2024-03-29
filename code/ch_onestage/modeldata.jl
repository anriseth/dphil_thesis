using Distributions

abstract type AbstractUncertaintyModel end

type ModelData
    a::Array{Float64}
    B::Array{Float64,2}
end

function data3d()
    a3d = [1.,.9, 1.2]
    B3d = [2 2 0; 0.8 1.8 8; 3 0 2]

    μ3d = [0.5, 0.5, 0.65]
    std3d = 0.1μ3d
    corr3d = [1 -0.3 0; -0.3 1 0; 0 0 1]
    @assert corr3d == transpose(corr3d)
    Σ3d = corr3d .* (std3d * std3d')

    locy3d = location(MvLogNormal, :meancov, μ3d, Σ3d)
    scaley3d = scale(MvLogNormal, :meancov, μ3d, Σ3d)

    return ModelData(a3d,B3d), MvLogNormal(locy3d, scaley3d)
end

function profitfun(x::Vector,y::Vector,data::ModelData)
    a = data.a
    B = data.B
    n = length(a)

    profit = sum((x[i]-y[i])*a[i]*exp(-B[i,i]*x[i])*
                 prod(1-exp(-x[j]*B[i,j]) for j=1:n if (i!=j) && !iszero(B[i,j]))
                 for i = 1:n)
end

function profitsample(x,y::Distribution,data::ModelData)
    profitfun(x, rand(y), data)
end
