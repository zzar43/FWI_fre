include("model_parameter.jl");
include("scalar_helmholtz_solver.jl");
include("FWI_fre.jl");
using JLD2, PyPlot;


@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all");

@save "data_compute/three_layers_data.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,8,3],Nx,Ny))); colorbar(); savefig("wavefield.png")
