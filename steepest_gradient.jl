addprocs(2)

include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
using JLD2, PyPlot;

# ================================================
# Read recorded data
@load "data_compute/three_layers_data.jld2" wavefield_true recorded_data_true
# ================================================

iter_time = 5;
c = 1e-5;
tau = 0.5;
search_time = 7;
misfit_diff_vec = zeros(iter_time);
alpha0 = 10;

fre_range = [1:2,1:2,3:4,3:4,5:6,5:6,7:9,7:9,7:9];
for iter_main = 1:iter_time
    println("=======================================")
    println("Main iteration time: ", iter_main)
    # Compute gradient
    @time gradient, misfit_diff = compute_gradient_parallel(vel_init, recorded_data_true, source_multi, acq_fre, fre_range[iter_main]);
    filename = "gradient_$iter_main.npy";
    npzwrite(filename, gradient)
    # Compute direction
    p = -gradient./norm(gradient);
    # line search
    alpha = backtracking_line_search_parallel(vel_init,p,gradient,alpha0,misfit_diff,tau,c,search_time,fre_range[iter_main],recorded_data_true);
    # update velocity
    vel_init = vel_init + alpha * p;
end

matshow(real(vel_init'), clim=[2,3]); colorbar()
matshow(real(vel_true'), clim=[2,3]); colorbar()
