# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");
@everywhere include("forward_modelling.jl");
# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);

# vel_new = steepest_gradient(vel_init, conf, recorded_data, vmin, vmax; alpha_1=30, alpha_max=500, iter_time=10, c1=1e-11, c2=0.9, search_time=5, zoom_time=5, verbose=true, save_graph=true, fre_range="all");
#
# @save "result_sd.jld2" vel_new

# vel_new = conjugate_gradient(vel_init, conf, recorded_data, vmin, vmax; alpha_1=30, alpha_max=500, iter_time=10, c1=1e-11, c2=0.9, search_time=5, zoom_time=6, verbose=true, save_graph=true, fre_range="all");

# @save "result_cg.jld2" vel_new

vel_new = l_BFGS(vel_init, conf, recorded_data, vmin, vmax; m=3, alpha_1=10, alpha_max=500, iter_time=10, c1=1e-11, c2=0.9, search_time=7, zoom_time=5, verbose=true, save_graph=true, fre_range="all");

matshow((reshape(vel_new,conf.Nx,conf.Ny))',cmap="PuBu"); colorbar()
@save "result_lbfgs.jld2" vel_new

matshow(vel_true',cmap="PuBu"); colorbar()

A = ones(3,4);

b = ones(3,1)

for i = 1:-1:1
	print(i)
end
