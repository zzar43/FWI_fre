using HDF5, JLD2

n, d, o, m0, m = read(h5open("overthrust_model_2D.h5", "r"), "n", "d", "o", "m0", "m");

Nx = n[1];
Ny = n[2];
h = 25;
vel_true = m;
vel_init = m0;

@save "data/overthrust.jld2" vel_true vel_init Nx Ny h
