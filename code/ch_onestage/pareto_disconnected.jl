#==
Example taken from
Messac et. al., Amir Ismail-Yahaya, and Christopher A Mattson.
The normalized normal constraint method for generating the pareto frontier.
Structural and multidisciplinary optimization, 25(2):86–98, 2003.

This example shows that the NBI method will generate local Pareto points that
are not global and dominated points that are not locally optimal.
We can make sure all NBI points are locally Pareto optimal by using an
inequality-constrained extension.
==#

using MultiJuMP, JuMP
using Ipopt
using JLD2

savedata = true
savedir = "./data/"
method = :EPS # or :EPS or :WS
ineq = true
methodmap = Dict(:NBI=>"nbi_eq", :WS=>"ws", :EPS=>"eps")
if ineq == true
    methodmap[:NBI] = "nbi_ineq"
end
pointsperdim = 40

m = MultiModel(solver = IpoptSolver(print_level=0))
@variable(m, 0 <= x[i=1:2] <= 5)
@NLexpression(m, f1, x[1])
@NLexpression(m, f2, x[2])
@NLconstraint(m, 5exp(-x[1])+2exp(-0.5(x[1]-3)^2) <= x[2])

obj1 = SingleObjective(f1, sense = :Min)

# As the problem is nonconvex, we have to supply the
# initial value to get the global minimum of f2
obj2 = SingleObjective(f2, sense = :Min,
                       iv = Dict{Symbol, Any}(:x => [5., 0.]))

multim = getMultiData(m)
multim.objectives = [obj1, obj2]
multim.pointsperdim = pointsperdim

if method == :NBI
    solve(m, method = method, inequalityconstraint = ineq)
else
    solve(m, method = method)
end

f1arr = convert(Array{Float64},
                [val[1] for val in multim.paretofront])
f2arr = convert(Array{Float64},
                [val[2] for val in multim.paretofront])
methodname = methodmap[method]
if savedata === true
    @save(savedir*"pareto_disconnected_$(methodname).jld2", f1arr, f2arr)
    writedlm(savedir*"pareto_disconnected_$(methodname).dat", zip(f1arr, f2arr))
end
