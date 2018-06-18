include("model_parameter_fre.jl");
include("scalar_helmholtz_solver.jl");
include("FWI_fre.jl");
using NPZ;
using PyPlot;

# ================================================
# Make recorded data
@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all");
npzwrite("data_wavefield_true.npy", wavefield_true)
npzwrite("data_recorded_true.npy", recorded_data_true)
# Read recorded data
wavefield_true = npzread("data_wavefield_true.npy");
recorded_data_true = npzread("data_recorded_true.npy");
matshow(real(reshape(wavefield_true[:,3,4],Nx,Ny)'), clim=[-2,2], cmap="seismic"); colorbar()
plot(real(recorded_data_true[1:150,2,1]))
# ================================================

iter_time = 5;
c = 1e-5;
tau = 0.5;
search_time = 7;
misfit_diff_vec = zeros(iter_time);
alpha0 = 10;

fre_range = [1:2,1:2,3:4,3:4,5];
for iter_main = 1:iter_time
    println("=======================================")
    println("Main iteration time: ", iter_main)
    # Compute gradient
    @time gradient, misfit_diff = compute_gradient(vel_init, recorded_data_true, source_multi, acq_fre, fre_range[iter_main]);
    filename = "gradient_$iter_main.npy";
    npzwrite(filename, gradient)
    # Compute direction
    p = -gradient./norm(gradient);
    # line search
    alpha = backtracking_line_search(vel_init,p,gradient,alpha0,misfit_diff,tau,c,search_time,fre_range[iter_main],recorded_data_true);
    # update velocity
    vel_init = vel_init + alpha * p;
end

matshow(real(vel_init'), clim=[2,3]); colorbar()
matshow(real(vel_true'), clim=[2,3]); colorbar()
