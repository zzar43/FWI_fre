# addprocs(2)

include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
using JLD2, PyPlot;

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all");

@save "data_compute/marmousi_data.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,3,3],Nx,Ny)'), cmap="seismic", clim=[-2,2]); colorbar(); savefig("wavefield.png")
