# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init acq_fre
@load "data_compute/overthrust_small.jld2" recorded_data_true
vmin = minimum(vel_true);
vmax = maximum(vel_true);

# Source term
source_multi = build_source_multi(10,0.1,acq_fre,ricker=true);

# inverse
vel_sg, mis_sg = steepest_gradient(vel_init, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=15, iter_time=5, c=5e-8, tau=0.5, search_time=5, verbose=true, save_graph=true);

@save "overthrust_small_data.jld2" vel_sg mis_sg
