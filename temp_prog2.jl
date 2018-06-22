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
# 1 Hz
vel_1, mis_1 = steepest_gradient(vel_init, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.2, search_time=5, verbose=true, save_graph=true, single_fre=1);
vel_2, mis_2 = steepest_gradient(vel_1, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.2, search_time=5, verbose=true, save_graph=true, single_fre=1);

# 3 Hz
vel_3, mis_3 = steepest_gradient(vel_2, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=2);
vel_4, mis_4 = steepest_gradient(vel_3, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=2);
vel_5, mis_5 = steepest_gradient(vel_4, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=10, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=2);

# 5 Hz
vel_5, mis_5 = steepest_gradient(vel_5, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=15, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=3);
vel_6, mis_6 = steepest_gradient(vel_5, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=3);

# 7 Hz
vel_7, mis_7 = steepest_gradient(vel_6, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=15, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=4);
vel_8, mis_8 = steepest_gradient(vel_7, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=4);

# 9 Hz
vel_9, mis_9 = steepest_gradient(vel_8, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=100, iter_time=5, c=1e-9, tau=0.3, search_time=5, verbose=true, save_graph=true, single_fre=5);

matshow((vel_init)');colorbar()
matshow((vel_2)');colorbar()
matshow((vel_3 - vel_4)');colorbar()


@save "overthrust_small_data.jld2" vel_1 vel_2 vel_3 vel_4 vel_5 vel_6 vel_7 vel_8
