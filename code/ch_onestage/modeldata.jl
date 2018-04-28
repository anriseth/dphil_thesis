using Distributions

abstract type AbstractUncertaintyModel end

type ModelData
    a::Array{Float64}
    B::Array{Float64,2}
end

function data3d()
    a3d = [1.,.9, 1.2]
    B3d = [2 10 0; 4 1.8 40; 15 0 2]

    μ3d = [0.5, 0.5, 0.65]
    std3d = 0.1μ3d
    corr3d = [1 -0.3 0; -0.3 1 0; 0 0 1]
    @assert corr3d == transpose(corr3d)
    Σ3d = corr3d .* (std3d * std3d')

    locy3d = location(MvLogNormal, :meancov, μ3d, Σ3d)
    scaley3d = scale(MvLogNormal, :meancov, μ3d, Σ3d)

    return ModelData(a3d,B3d), MvLogNormal(locy3d, scaley3d)
end
