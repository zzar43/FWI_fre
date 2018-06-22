addprocs(2);
@everywhere include("forward_modelling.jl");
@everywhere include("inverse_problem.jl");
using JLD2, PyPlot;

# Load data
@load "data/three_layers.jld2" vel_true vel_init acq_fre
@load "data_compute/three_layers.jld2" recorded_data_true

# Source term
source_multi = build_source_multi(10,0.1,acq_fre,ricker=true);

# Gradient
grad = compute_gradient(vel_init, source_multi, acq_fre, [3], recorded_data_true, verbose=true);
p = -grad./maximum(abs.(grad));

matshow(p', cmap="seismic", clim=[-1,1]); colorbar()

alpha, mis = backtracking_line_search(vel_init,source_multi,acq_fre,p,grad,recorded_data_true,2,3,10,0.5,0.5,5,"all",verbose=true);
