# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");
@everywhere include("forward_modelling.jl");
# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);

grad = compute_gradient(vel_init, conf, recorded_data; fre_range=[1], verbose=true);

matshow(grad')

Nx_pml = conf.Nx + 2*conf.pml_len;
Ny_pml = conf.Ny + 2*conf.pml_len;
omega = 2*pi*conf.frequency;
gradient = SharedArray{Float32}(Nx_pml*Ny_pml, conf.fre_num, conf.source_num);
# recorded_forward = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
misfit_diff = 0.0;
# Receiver projector
R1 = build_proj_op1(conf);
R2 = build_proj_op2(conf);

ind_fre = 1; ind_source = 1;
vel = vel_init;

A = make_diff_operator(vel,conf,ind_fre=ind_fre);
F = lufact(A);

source = make_source(conf,ind_fre=ind_fre,ind_source=ind_source);
# Forward
u_forward_vec = F\source; # size [Nx_pml*Ny_pml,1]

matshow(real(reshape(u_forward_vec,Nx_pml,Ny_pml))')
# Adjoint source
r_forward_vec = R2 * u_forward_vec; # size [Nx_pml*Ny_pml,1]
r = R1 * u_forward_vec;
plot(real(r))
plot(real(recorded_data[:,ind_fre,ind_source]))
source_adjoint = -1 * (r_forward_vec - R1.'*recorded_data[:,ind_fre,ind_source]);
plot(real(r_forward_vec - R1.'*recorded_data[:,ind_fre,ind_source]))
u_back_vec = F\source_adjoint; # size [Nx_pml*Ny_pml,1]
matshow(real(reshape(u_back_vec,Nx_pml,Ny_pml))')
grad = real(omega[ind_fre].^2 .* conj(u_forward_vec) .* u_back_vec);
matshow(real(reshape(grad,Nx_pml,Ny_pml))')
grad = reshape(grad,Nx_pml,Ny_pml);
grad = grad[conf.pml_len+1:end-conf.pml_len,conf.pml_len+1:end-conf.pml_len];

matshow(grad')
