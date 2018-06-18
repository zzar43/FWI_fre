addprocs(2)

include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
using JLD2, PyPlot;

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver_parallel(vel_true, source_multi, acq_fre, "all");

@save "data_compute/three_layers_data.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,1,2],Nx,Ny)'), cmap="seismic", clim=[-2,2]); colorbar(); savefig("wavefield.png")
