@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("def_structure.jl");
@everywhere include("compute_gradient.jl");
@everywhere include("optimization/optimization.jl");
@everywhere include("optimization/line_search.jl");
using JLD2, PyPlot;
@load "data/overthrust_small.jld2" vel_true vel_init
@load "data_compute/overthrust_small.jld2" recorded_data_true


vmax = maximum(vel_true);
vmin = minimum(vel_true);

@time grad = compute_gradient_parallel(vel_init, acq_fre, [3], recorded_data_true, true);
p = -grad / maximum(abs.(grad));
matshow(p', cmap="seismic", clim=[-0.5,0.5]); colorbar()


alpha = 15;
vel_new = update_velocity(vel_init,alpha,p,vmin,vmax);
matshow(vel_new', cmap="seismic"); colorbar()
matshow(vel_true', cmap="seismic"); colorbar()

j0 = compute_misfit_func(vel_init, acq_fre, recorded_data_true, [3])
j1 = compute_misfit_func(vel_new, acq_fre, recorded_data_true, [3])
matshow((vel_new-vel_init)', cmap="jet"); colorbar()

@time grad1 = compute_gradient_parallel(vel_new, acq_fre, [1], recorded_data_true, true);
p1 = -grad1 / maximum(abs.(grad1));
matshow(p1', cmap="seismic", clim=[-0.05,0.05]); colorbar()

alpha = 100000;
vel_new1 = update_velocity(vel_new,alpha,p1,vmin,vmax);
matshow(vel_new1', cmap="jet"); colorbar()
matshow(vel_init', cmap="jet"); colorbar()


vel_new, misfit_vec = steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax; alpha0=1000, iter_time=5, c=1e-3, tau=0.4, search_time=5, verbose=true, save_graph=true);

matshow(vel_new', cmap="seismic"); colorbar()
