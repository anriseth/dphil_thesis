using JuMP, MultiJuMP
using Ipopt # Use Ipopt as the underlying nonlinear solver
# Define parameters
α     = [1, 1.5, 1.2]
β     = [-2 0.23 0; 0.15 -2.1 0.04; 0.1 0 -3.]
μy    = [1.5, 1.2, 1.3]
σy    = [0.17, 0.2, 0.1].*μy
corry = [1 0.5 0; 0.5 1 0 ; 0 0 1]
covy  = corry .* (σy * σy')
x0    = [1.5, 2.0, 1.7].*μy
n     = length(x0)

# Set up Multiobjective model
m = MultiModel(solver = IpoptSolver())

# JuMP commands create functions
@variable(m, x[i=1:n] >= 0, start=x0[i])
@NLexpressions m begin
    demand[i=1:n],  α[i]*prod(x[k]^β[i,k] for k=1:n)
    profit[i=1:n],  (x[i]-μy[i])*demand[i]
    totalProfit,    sum(profit[i] for i=1:n)
    stdeviation,    sqrt(sum(covy[i,j]*demand[i]*demand[j]
                             for j=1:n for i=1:n))
end
@constraint(m, moveconstr[i=1:n], 0.80x0[i] <= x[i] <= 1.20x0[i])

# Define two objectives
md              = getMultiData(m)
obj1            = SingleObjective(stdeviation, sense = :Min)
obj2            = SingleObjective(totalProfit, sense = :Max)
md.objectives   = [obj1, obj2]
md.pointsperdim = 30

# solve calls MultiJuMP to approximate Pareto front
solve(m, method = :NBI)

# The Pareto points are stored in md.paretofront
