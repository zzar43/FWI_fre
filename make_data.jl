# addprocs(2)
using JLD2, PyPlot;

# include functions
@everywhere include("forward_modelling.jl")

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf

# Forward modelling
@time wavefield_true, recorded_data = scalar_helmholtz_solver(vel_true, conf; fre_range="all", verbose=true);

# @code_warntype scalar_helmholtz_solver(vel_true, conf; verbose=true)

@save "data_compute/overthrust_small.jld2" recorded_data

matshow(real(wavefield_true[:,:,5,20])',cmap="RdBu",clim=[-2,2]); colorbar();
savefig("temp_graph/wavefield.png")
plot(real(recorded_data[:,6,1]))
