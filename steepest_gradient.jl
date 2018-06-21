addprocs(2)

# include("model_parameter.jl");
using JLD2, PyPlot
# @everywhere include("model_func.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("compute_gradient.jl");
@everywhere include("def_structure.jl");
@everywhere include("optimization/optimization.jl")
@everywhere include("optimization/line_search.jl")

# ================================================
# Read data
@load "data/marmousi.jld2" vel_true vel_init acq_fre;
@load "data_compute/marmousi.jld2" wavefield_true recorded_data_true;
matshow((vel_true)', cmap="plasma"); colorbar(); savefig("temp_graph/vel_true.png")
matshow((vel_init)', cmap="plasma"); colorbar(); savefig("temp_graph/vel_init.png")
# ================================================

vmin = minimum(vel_true);
vmax = maximum(vel_true);


iter_time = 10;
alpha0 = 0.1*maximum(vel_true);
search_time = 6;
c = 1e-4;
tau = 0.3;

vel_init, misfit_vec = steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax; alpha0=alpha0, iter_time=iter_time, c=c, tau=tau, search_time=search_time, verbose=true, save_graph=true);


clf()
gcf()
plot(misfit_vec)

# # For three layers
# c = 1e-5;
# tau = 0.3;
# search_time = 4;
# alpha0 = 8;
# vmin = 2; vmax = 3;
#
# iter_time = 3;
# fre_range = [1:1, 1:1, 2:2, 2:2, 2:2, 2:2, 3:3, 3:3, 3:3, 3:3, 4:4, 4:4, 4:4, 5:5, 5:5];
# misfit_vec = zeros(iter_time);
#
# for iter_main = 1:iter_time
#     println("\nSteepest Gradient Iteration: ", iter_main)
#     # Compute gradient
#     grad = compute_gradient_parallel(vel_init, acq_fre, fre_range[iter_main], recorded_data_true,false);
#     p = -grad / norm(grad);
#     alpha, misfit_value = backtracking_line_search(vel_init,acq_fre,p,grad,recorded_data_true,vmin,vmax,alpha0,tau,c,search_time,"all",false)
#     vel_init = update_velocity(vel_init,alpha,p,vmin,vmax);
#     misfit_vec[iter_main] = misfit_value;
#     matshow((vel_init)');colorbar();savefig("temp_graph/vel_$iter_main.png")
# end
#
# matshow(vel_init.', cmap="gray")
#
# plot(1:15,misfit_vec)
#
#
# vel_init, misfit_vec = steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax, 8, 10, 1e-5, 0.5, 4, "all", true, true);
