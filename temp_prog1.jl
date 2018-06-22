addprocs(2);
@everywhere include("forward_modelling.jl");
@everywhere include("inverse_problem.jl");
using JLD2, PyPlot;

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init acq_fre
@load "data_compute/overthrust_small.jld2" recorded_data_true

# Source term
source_multi = build_source_multi(10,0.1,acq_fre,ricker=true);

# Gradient
grad = compute_gradient(vel_init, source_multi, acq_fre, [3], recorded_data_true, verbose=true);
p = -grad./maximum(abs.(grad));

matshow(p', cmap="RdBu", clim=[-1,1]); colorbar()

vmin = minimum(vel_true);
vmax = maximum(vel_true);
alpha = 10;
tau = 0.5;
c = 1e-6;
search_time = 5;
fre_range = "all";
alpha, mis = backtracking_line_search(vel_init,source_multi,acq_fre,p,grad,recorded_data_true,vmin,vmax,10,0.5,0.5,5,"all",verbose=true);
backtracking_line_search(vel,source_multi,acq_fre,p,gradient,recorded_data,vmin,vmax,alpha,tau,c,search_time,fre_range;verbose=true)
