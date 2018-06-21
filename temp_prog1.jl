include("scalar_helmholtz_solver.jl");
include("def_structure.jl");
include("compute_gradient.jl");
include("optimization/optimization.jl");
include("optimization/line_search.jl");
using JLD2, PyPlot;
@load "data/marmousi.jld2" vel_true vel_init acq_fre;
@load "data_compute/marmousi.jld2" recorded_data_true;

#
#
# @time grad = compute_gradient_parallel(vel_init, acq_fre, [6], recorded_data_true, true);
#
# p = -grad / maximum(abs.(grad));
# matshow(p', cmap="seismic", clim=[-0.01,0.01]); colorbar()
#
# alpha = 10000;
# vel_new = update_velocity(vel_init,alpha,p,vmin,vmax);
# matshow(vel_new', cmap="seismic"); colorbar()
vmax = maximum(vel_true);
vmin = minimum(vel_true);
vel_new, misfit_vec = steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax; alpha0=1000, iter_time=5, c=1e-3, tau=0.4, search_time=5, verbose=true, save_graph=true);

matshow(vel_new', cmap="seismic"); colorbar()
