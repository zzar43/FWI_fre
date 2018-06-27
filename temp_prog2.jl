# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");
@everywhere include("forward_modelling.jl");
# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" wavefield_true recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);

vel_new = steepest_gradient(vel_init, conf, recorded_data, vmin, vmax; alpha_1=100, alpha_max=500, iter_time=5, c1=1e-11, c2=0.9, search_time=5, zoom_time=5, verbose=true, save_graph=true, fre_range="all");

vel_new = conjugate_gradient(vel_init, conf, recorded_data, vmin, vmax; alpha_1=100, alpha_max=500, iter_time=5, c1=1e-11, c2=0.9, search_time=5, zoom_time=6, verbose=true, save_graph=true, fre_range="all");
