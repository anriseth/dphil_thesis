using JuMP
using Optim
using Plots
pgfplots()

include("meanutilmodel.jl")
include("meandevmodel.jl")
include("cvarmodel.jl")


function optimalx(λ, umodel::AbstractUncertaintyModel)
    setvalue(umodel.λ, λ)
    status = solve(umodel.model)
    @assert status == :Optimal

    return getvalue(umodel.x)
end

function Dp(λ1, λ2, model1, model2, p = 2)
    x1 = optimalx(λ1, model1)
    x2 = optimalx(λ2, model2)
    return vecnorm(x1-x2, p)
end

function nearestλ(l, lmodel::AbstractUncertaintyModel,
                  λguess, λmodel::AbstractUncertaintyModel;
                  lower::Real=0., upper::Real=3.)
    # Given parameter l for lmodel, find the λ for λmodel that
    # gives the smallest 2-norm of optimalx between λmodel and
    # lmodel
    tol = 1e-3

    x1 = optimalx(l, lmodel)
    function f(λ)
        # Update initial value in case
        # the optimisation algorithm gets stuck in a local optima
        #setvalue(λmodel.x, x1)
        sum(abs2, x1-optimalx(λ[1], λmodel))
    end
    res = optimize(f, [λguess], LBFGS(), Optim.Options(g_tol = 1e-10))
    fmin = sqrt(Optim.minimum(res) / sum(abs2, x1))

    #λsol = Optim.minimizer(res)[1]
    #xl = vecnorm(x1)
    #xλ = vecnorm(getvalue(λmodel.x))
    #@show(λguess, λsol, λsol-λguess, fmin, xl, xλ)
    @assert Optim.converged(res)
    xmin = Optim.minimizer(res)[1]
    if xmin > upper*(1-tol)
        Base.warn("xmin > upper*(1-tol):\t$xmin > $(upper*(1-tol))")
    end
    if xmin < lower*(1+tol)
        Base.warn("xmin < lower*(1+tol):\t$xmin < $(lower*(1+tol))")
    end

    return (xmin, fmin)
end

modeldata3d, y3d = data3d() # modeldata.jl
modelmeanutil3d = MeanUtilityModel(modeldata3d, y3d, N = Int(1e5))
modelcvar3d = CvarModel(modeldata3d, y3d, N = Int(1e5))
modelmeandev3d = MeanDeviationModel(modeldata3d, mean(y3d), cov(y3d))

function expoutildist(μs, λguess)
    numvals = length(μs)
    λs = zeros(numvals)
    utilmin = zeros(numvals)
    Base.info("μ = $(μs[1])")
    λs[1], utilmin[1] = nearestλ(μs[1], modelmeanutil3d,
                                 λguess, modelmeandev3d,
                                 lower=0., upper=3.)
    for i in 2:numvals
        Base.info("μ = $(μs[i])")
        λguess = λs[i-1]
        λs[i], utilmin[i] = nearestλ(μs[i], modelmeanutil3d,
                                     λguess, modelmeandev3d,
                                     lower=0.5*λs[i-1], upper=3)
    end
    return λs, utilmin
end


function cvardist(γs, λguess)
    numvals = length(γs)
    λs = zeros(numvals)
    cvarmin = zeros(numvals)
    Base.info("γ = $(γs[1])")
    λs[1], cvarmin[1] = nearestλ(γs[1], modelcvar3d,
                                 λguess, modelmeandev3d,
                                 lower=0., upper=1e-1)
    for i in 2:numvals
        Base.info("γ = $(γs[i])")
        λguess = λs[i-1]
        λs[i], cvarmin[i] = nearestλ(γs[i], modelcvar3d,
                                     λguess, modelmeandev3d,
                                     lower=0.5*λs[i-1], upper=3)
    end
    return λs, cvarmin
end

function plotdiffs(x, y, z,
                   xlabel, ylabel, zlabel,
                   saveimg, fname, upscale)

    plt1 = plot(x, z, xlabel=xlabel, ylabel=zlabel, label="")
    plt2 = plot(x, y, xlabel=xlabel, ylabel=ylabel, label="")

    if saveimg
        savefig(plt1, fname*"_x.tex")
        savefig(plt2, fname*"_l.tex")
        #plt = plot(plt1, plt2, layout=(1,2),
        #           size=(800*upscale, 300*upscale))
        #savefig(plt, fname)
    end
    return plt1, plt2
end

function plotutilmin(μs, λs, xmin, saveimg = true,
                     fname = "./data/util_meandev_3d",
                     upscale = 1.5)
    plotdiffs(μs, λs, xmin,
              "\$\\mu\$", "\$\\lambda\$", "\$\\|x(\\mu)-x(\\lambda)\\| / \\|x(\\mu)\\|\$",
              saveimg, fname, upscale)
end

function plotcvarmin(γs, λs, xmin, saveimg = true,
                     fname = "./data/cvar_meandev_3d",
                     upscale = 1.5)
    plotdiffs(γs, λs, xmin,
              "\$\\gamma\$", "\$\\lambda\$", "\$\\|x(\\gamma)-x(\\lambda)\\| / \\|x(\\gamma)\\|\$",
              saveimg, fname, upscale)
end

function xchanges(λ, model::AbstractUncertaintyModel)
    # Let λ[1] represent ≈ risk-neutral
    numvals = length(λ)
    xvals = zeros(2, numvals)

    for i in 1:length(λ)
        xvals[:, i] = optimalx(λ[i], modelmeandev3d)
    end

    xdist = [norm(xvals[:,i]-xvals[:,1])/norm(xvals[:,1])
             for i in 1:size(xvals,2)]
    return convert(Array{Float64}, xdist)
end


# Find (λ,μ)-pairs that generate the same solution
μvals = 163
μs = linspace(1, 82, μvals)
λguess = 0.0
λsutil, utilmin = expoutildist(μs, λguess)
pltdistu, pltlu = plotutilmin(μs, λsutil, utilmin, true)



# Find (λ,γ)-pairs that generate the same solution
γvals = 60
γs = linspace(1, 1e-2, γvals)
λguess = 0.0
λscvar, cvarmin = cvardist(γs, λguess)
pltdistc, pltlc = plotcvarmin(γs, λscvar, cvarmin, true)


#==
# Investigate the relative change in the solution
# when we change the parameter
λvals = 50
λs = linspace(0,2,λvals)
xreldist = xchanges(λs, modelmeandev3d)

pltchange = plot(x=λs, y=xreldist, Geom.line,
Guide.xlabel("λ"),
Guide.ylabel("Relative 2-norm"),
Guide.title("Relative solution distance compared to risk-neutral"))

draw(PS("./data/relchange_meandev.eps", 4inch, 3inch), pltchange)
==#
