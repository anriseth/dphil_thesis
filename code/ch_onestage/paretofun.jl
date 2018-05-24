using JuMP, MultiJuMP, Ipopt
using Distributions
using Plots
using JLD2

varprof = false
saveimg = true
savedata = true
savedir = "./data/"
method = :NBI # or :EPS or :WS
methodmap = Dict(:NBI=>"nbi", :WS=>"ws", :EPS=>"eps")

include("modeldata.jl")
data, ydist = data3d()

a = data.a
B = data.B
n = length(a)

μy = mean(ydist)
Cy = cov(ydist)
x0 = 2μy
@show (μy, Cy, x0)

m = MultiModel(solver = IpoptSolver(max_iter = 100))
@variable(m, x[i=1:n] >= 0, start=x0[i])
@NLexpressions m begin
    demand[i=1:n],  a[i]*exp(-B[i,i]*x[i])*prod(1-exp(-B[i,j]*x[j]/x[i]) for j=1:n if (j != i) && !iszero(B[i,j]))
    profit[i=1:n],  (x[i]-μy[i])*demand[i]
    totalProfit,    sum(profit[i] for i=1:n)
    stdeviation,    sqrt(sum(Cy[i,j]*demand[i]*demand[j] for j=1:n for i=1:n))
    revenue[i=1:n], x[i]*demand[i]
    totalRevenue,   sum(revenue[i] for i=1:n)
end


multim = getMultiData(m)
if varprof === true
    multim.pointsperdim = 30
else
    multim.pointsperdim = 30
end

function plotfunc(z, fun)
    setValue(x,z)
    getValue(fun)
end

if varprof === true
    obj1 = SingleObjective(stdeviation, sense = :Min)
    obj2 = SingleObjective(totalProfit, sense = :Max)
    multim.objectives = [obj1, obj2]

    solve(m, method = method)

    f1arr = convert(Array{Float64},
                    [val[1] for val in multim.paretofront])
    f2arr = convert(Array{Float64},
                    [val[2] for val in multim.paretofront])

    profits = f2arr
    σprofits = f1arr

    stdprof = plot(getMultiData(m))

    methodname = methodmap[method]
    if saveimg === true
        savefig(stdprof, savedir*"pareto_std_prof_3_$(methodname).pdf")
    end
    if savedata === true
        @save(savedir*"pareto_std_prof_3_$(methodname).jld2", profits, σprofits)
        writedlm(savedir*"pareto_std_prof_3_$(methodname).dat", zip(σprofits, profits))
    end
else
    obj1 = SingleObjective(totalRevenue, sense = :Max)
    obj2 = SingleObjective(totalProfit, sense = :Max)
    multim.objectives = [obj1, obj2]

    solve(m, method = method)

    f1arr = convert(Array{Float64},
                    [val[1] for val in multim.paretofront])
    f2arr = convert(Array{Float64},
                    [val[2] for val in multim.paretofront])

    profrev = plot(getMultiData(m), xlabel="Revenue", ylabel="Expected Profit")

    methodname = methodmap[method]

    if saveimg === true
        savefig(profrev, savedir*"pareto_prof_rev_$(methodname).pdf")
    end
    if savedata === true
        @save(savedir*"pareto_prof_rev_$(methodname).jld2", f1arr, f2arr)
        writedlm(savedir*"pareto_prof_rev_$(methodname).dat", zip(f1arr, f2arr))
    end
end
