
# Building model
include("def_structure.jl")
include("model_func.jl")

# Gradient
include("compute_gradient.jl")

include("optimization/conjugate_gradient.jl")
include("optimization/line_search.jl")
include("optimization/steepest_gradient.jl")
include("optimization/l_BFGS.jl")
