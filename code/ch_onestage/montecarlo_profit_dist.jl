using StatPlots
pgfplots()
include("modeldata.jl")

data, y = data3d()
N = Int(5e6)
xarr = [[1,1,1.0], [0.8,1,0.9], [1.3,1.1, 1.0]]

function getsamples(x, y::Distribution, data::ModelData, N::Int)
    # Make this a pmap
    info("Generating sample using x = $x.")
    [profitsample(x, y, data) for i = 1:N]
end

function createplots(xarr, montedists::Array, N::Int,
                     saveimg::Bool=true; savefile = "./data/montecarlo_profit_dist.tex")
end

function createplots(xarr, y::Distribution, data::ModelData, N::Int,
                     saveimg::Bool=true; savefile = "./data/montecarlo_profit_dist.tex")

    numplts = length(xarr)
    montedists = [getsamples(xarr[i], y, data, N) for i = 1:length(xarr)]
    plt = density(montedists[1])
    for j = 2:numplts
        density!(plt, montedists[j])
    end
    plt = plot(plt, xlabel="\$f(x,y)\$", ylabel="Density",
               xlim=(0.03, 0.085),
               legend = :topleft,
               labels = reshape(["\$x=$(string(xarr[j]))\$"
                                 for j=1:numplts], 1,numplts))

    if saveimg == true
        savefig(plt, savefile)
    end

    plt, montedists
end

srand(0) # Set seed

plt, montedists = createplots(xarr, y, data, N)
