# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");
@everywhere include("forward_modelling.jl");
# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" wavefield_true recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);

grad = compute_gradient(vel_init, conf, recorded_data; fre_range=[5], verbose=true);

matshow(grad'); colorbar()

Nx_pml = conf.Nx + 2*conf.pml_len;
Ny_pml = conf.Ny + 2*conf.pml_len;
omega = 2*pi*conf.frequency;
gradient = SharedArray{Float32}(Nx_pml*Ny_pml, conf.fre_num, conf.source_num);
# recorded_forward = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
misfit_diff = 0.0;
# Receiver projector
R1 = build_proj_op1(conf);
R2 = build_proj_op2(conf);

ind_fre = 5; ind_source = 20;
vel = vel_init;
A = make_diff_operator(vel,conf,ind_fre=ind_fre);
F = lufact(A);
source = make_source(conf,ind_fre=ind_fre,ind_source=ind_source);


u_forward_vec = F\source; # size [Nx_pml*Ny_pml,1]
r_forward_vec = R2 * u_forward_vec; # size [Nx_pml*Ny_pml,1]

matshow(real(reshape(r_forward_vec,Nx_pml,Ny_pml))');

source_adjoint = -1 * (r_forward_vec - R1.'*recorded_data[:,ind_fre,ind_source]);
plot(real(recorded_data[:,ind_fre,ind_source]))
matshow(real(reshape(source_adjoint,Nx_pml,Ny_pml))');

u_back_vec = F\source_adjoint; # size [Nx_pml*Ny_pml,1]
u_back = reshape(u_back_vec,Nx_pml,Ny_pml); matshow(real(u_back'));
grad = real(omega[ind_fre].^2 .* conj(u_forward_vec) .* u_back_vec);
grad = reshape(grad,Nx_pml,Ny_pml);
matshow(grad')
plot(real(R1*source_adjoint))

A = make_diff_operator(vel,conf,ind_fre=ind_fre);
B =  make_diff_operator(vel_true,conf,ind_fre=ind_fre);

source = make_source(conf,ind_fre=ind_fre,ind_source=ind_source);


u1 = A\source;
u2 = B\source;
r1 = R1 * u1;
r2 = R1 * u2;

plot(r2)

plot(real(r1-r2))

u11 = reshape(u1,Nx_pml,Ny_pml);
u21 = reshape(u2,Nx_pml,Ny_pml);
matshow(real(u11)');colorbar()
matshow(real(u21)');colorbar()

source_adj = R1 * (u1-u2);
plot(real(source_adj))
source_adj = R1.' * source_adj;

u3 = A\source_adj;
u31 = reshape(u3,Nx_pml,Ny_pml);
matshow(real(u31)');colorbar()

grad = real(conj(u11) .* u31);
matshow(grad')
