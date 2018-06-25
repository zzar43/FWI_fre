addprocs(2)
using JLD2, PyPlot;

# include functions
@everywhere include("forward_modelling.jl")

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf

# Forward modelling
@time wavefield_true, recorded_data = scalar_helmholtz_solver(vel_init, conf; fre_range="all", verbose=true);

@save "data_compute/overthrust_small.jld2" wavefield_true recorded_data

matshow(real(wavefield_true[:,:,7,40])',cmap="RdBu",clim=[-0.2,0.2]); colorbar();
savefig("temp_graph/wavefield.png")
