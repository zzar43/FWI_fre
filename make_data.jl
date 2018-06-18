include("model_parameter.jl");
include("scalar_helmholtz_solver.jl");
include("FWI_fre.jl");
using JLD2, PyPlot;

matshow(real(vel_true')); colorbar()

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all");

@save "data_compute/wavefield_data.jld2" wavefield_true recorded_data_true
