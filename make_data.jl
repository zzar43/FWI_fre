addprocs(2)
using JLD2, PyPlot;

# include functions
@everywhere include("forward_modelling.jl")

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init acq_fre

# Source term
source_multi = build_source_multi(10,0.1,acq_fre,ricker=true);

# Forward modelling
@time wavefield_true, recorded_data_true = scalar_helmholtz_solver(vel_true, source_multi, acq_fre, "all", verbose=true);

@save "data_compute/overthrust_small.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,7,5],acq_fre.Nx,acq_fre.Ny)'),cmap="RdBu",clim=[-2,2]); colorbar(); savefig("temp_graph/wavefield.png")
