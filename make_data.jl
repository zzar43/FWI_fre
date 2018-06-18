@everywhere include("model_parameter.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
@everywhere using JLD2, PyPlot;


@time wavefield_true, recorded_data_true = scalar_helmholtz_solver_parallel(vel_true, source_multi, acq_fre, "all");

@save "data_compute/three_layers_data.jld2" wavefield_true recorded_data_true

matshow(real(reshape(wavefield_true[:,8,3],Nx,Ny))); colorbar(); savefig("wavefield.png")
