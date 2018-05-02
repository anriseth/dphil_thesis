using JuMP, MultiJuMP
# Define parameters
a  = [1, 0.9, 1.2];
B  = [2 10 0; 4 1.8 40; 15 0 2]
μy = [0.5, 0.5, 0.65]
Cy = [0.0025 -0.00075 0; -0.00075 0.0025 0; 0 0 0.0042]
x0 = [1, 1, 1.3]
n  = length(x0)

# Set up Multiobjective model
m = MultiModel()

# JuMP commands create functions
@variable(m, x[i=1:n] >= 0, start=x0[i])
@NLexpressions m begin
    demand[i=1:n],  (a[i]*exp(-B[i,i]*x[i])*prod(1-exp(-x[j]*B[i,j])
                    for j=1:n if (j != i) && !iszero(B[i,j])))
    profit[i=1:n],  (x[i]-μy[i])*demand[i]
    totalProfit,    sum(profit[i] for i=1:n)
    stdeviation,    (sqrt(sum(Cy[i,j]*demand[i]*demand[j]
                              for j=1:n for i=1:n)))
end

# Define two objectives
md              = getMultiData(m)
obj1            = SingleObjective(stdeviation, sense = :Min)
obj2            = SingleObjective(totalProfit, sense = :Max)
md.objectives   = [obj1, obj2]
md.pointsperdim = 30

# solve calls MultiJuMP to approximate Pareto front
solve(m, method = :NBI)

# The Pareto points are stored in md.paretofront
