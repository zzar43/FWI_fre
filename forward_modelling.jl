
# Building model
include("def_structure.jl")
include("model_func.jl")

# Solver
include("scalar_helmholtz_solver.jl")

# Gradient
include("compute_gradient.jl")
