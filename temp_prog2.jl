# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");
@everywhere include("forward_modelling.jl");
# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" wavefield_true recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);

grad = compute_gradient(vel_init, conf, recorded_data; fre_range=[1], verbose=true);
@code_warntype compute_gradient(vel_init, conf, recorded_data; fre_range=[1], verbose=true)
p = -1 * grad / maximum(grad);

matshow(p'); colorbar()
