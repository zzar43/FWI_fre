# addprocs(2)

include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
using JLD2, PyPlot;

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all");

@save "data_compute/marmousi_data.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,1,1],Nx,Ny)'), cmap="seismic", clim=[-2,2]); colorbar(); savefig("wavefield.png")

@time gradient, misfit_diff = compute_gradient(vel_init, recorded_data_true, source_multi, acq_fre, "all");

matshow(gradient', cmap="seismic",clim=[-0.000001,0.000001]); colorbar(); savefig("gradient.png")
