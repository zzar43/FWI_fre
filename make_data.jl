# addprocs(2)

include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
using JLD2, PyPlot;
@load "data/marmousi.jld2" vel_true vel_init Nx Ny h source_multi acq_fre

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all");

@save "data_compute/marmousi.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,5,10],Nx,Ny)'), cmap="seismic"); colorbar(); savefig("temp_graph/wavefield.png")
