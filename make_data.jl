# addprocs(2)

# include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("def_structure.jl");
# @everywhere include("FWI_fre.jl");
using JLD2, PyPlot;
@load "data/overthrust_small.jld2" vel_true vel_init

@time wavefield_true, recorded_data_true = scalar_helmholtz_solver_parallel(vel_true, acq_fre, "all", 1:3);

@save "data_compute/overthrust_small.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,3,35],acq_fre.Nx,acq_fre.Ny)'),cmap="seismic"); colorbar(); savefig("temp_graph/wavefield.png")

u = reshape(wavefield_true[:,3,35],acq_fre.Nx,acq_fre.Ny);

matshow(real(u)', clim=[-1,1], cmap="seismic");colorbar()
matshow(real(u./(abs(u)))', clim=[-1,1], cmap="seismic");colorbar()
