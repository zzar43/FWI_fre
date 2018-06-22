addprocs(2);
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("def_structure.jl");
@everywhere include("compute_gradient.jl");
@everywhere include("model_func.jl");
@everywhere include("optimization/optimization.jl");
@everywhere include("optimization/line_search.jl");
using JLD2, PyPlot;

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init acq_fre
@load "data_compute/overthrust_small.jld2" recorded_data_true

# Source term
source_multi = build_source_multi(10,0.1,acq_fre,ricker=true);

# Gradient
grad = compute_gradient(vel_init, source_multi, acq_fre, fre_range, recorded_data; verbose=false);
p = -grad./maximum(abs.(grad));

matshow(p', cmap="seismic", clim=[-1,1])
