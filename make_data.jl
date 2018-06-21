# addprocs(10)

# include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("def_structure.jl");
# @everywhere include("FWI_fre.jl");
using JLD2, PyPlot;
@load "data/marmousi.jld2" vel_true vel_init acq_fre

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver_parallel(vel_true, acq_fre, "all", true);

@save "data_compute/marmousi.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,1,10],acq_fre.Nx,acq_fre.Ny)'), cmap="seismic", clim=[-1,1]); colorbar(); savefig("temp_graph/wavefield.png")
