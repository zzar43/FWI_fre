# Setup the model parameters.
using JLD2, PyPlot;
# Head function
include("model_func.jl");
include("def_structure.jl");

# Space
# @load "data/three_layers.jld2" vel_true vel_init acq_fre
# @load "data/marmousi.jld2" vel_true vel_init Nx Ny h
# @load "data/overthrust.jld2" vel_true vel_init acq_fre
@load "data/overthrust_small.jld2" vel_true vel_init

# using MAT;
# vars = matread("marmousi_dz10.mat");
# vel_true = vars["vel"]; vel_true = vel_true.';
# vel_true = convert(Array{Float32,2},vel_true)
# Nx, Ny = size(vel_true);
# h = 10;
# using ImageFiltering
# vel_init = imfilter(vel_true, Kernel.gaussian(15));

h = 25;
Nx = 401; Ny = 131;

# h = 0.01;
# Nx = 101; Ny = 101

# PML
pml_len = 50;
pml_alpha = 1;
Nx_pml = Nx + 2pml_len;
Ny_pml = Ny + 2pml_len;

# ===================================================
# Time
sample_fre = 1000.0; # Hertz
dt = 1/sample_fre;
Nt = 1000;
t = linspace(0,(Nt-1)*dt,Nt);
fre = sample_fre * linspace(0,1-1/Nt,Nt);
fre_position = 3:2:15;
frequency = fre[fre_position];
fre_num = length(frequency);
println("Frequency: ", frequency)

# ===================================================
# Source
source_num = 101;
source_coor = zeros(Int,source_num,2);
for i = 1:source_num
    source_coor[i,1] = 1 + 4*(i-1);
    # source_coor[i,1] = 201;
    source_coor[i,2] = 2;
end
ricker_func = source_ricker(8, 0.1, t);
source_value = ricker_func[fre_position]
# for i = 7:source_num
#     source_coor[i,1] = 1+(i-7)*20;
#     source_coor[i,2] = 101;
# end

println("Source number: ", source_num)
# source_multi = build_source_multi(10,0.1,t,fre_position,source_num,fre_num,true);
# ===================================================
# Receiver
receiver_num = Nx;
receiver_coor = zeros(Int,receiver_num,2);
for i = 1:receiver_num
    receiver_coor[i,1] = i;
    receiver_coor[i,2] = 2;
end
println("Receiver number: ", receiver_num)
# Projection operator
# R = build_proj_op(Nx,Ny,receiver_coor,receiver_num);
# R_pml = build_proj_op_pml(Nx,Ny,receiver_coor,receiver_num,pml_len);

# Display model
# draw_model(vel_true, vel_init, receiver_coor,source_coor);

# Make acquisition
# acq_fre = acquisition_fre(Nx,Ny,h,Nt,dt,t,frequency,fre_num,fre_position,source_num,source_coor,receiver_num,receiver_coor,pml_len,pml_alpha,Nx_pml,Ny_pml);

conf = configuration(Nx,Ny,h,Nt,dt,t,frequency,fre_num,fre_position,source_num,source_coor,source_value,receiver_num,receiver_coor,pml_len,pml_alpha)


@save "data/overthrust_small.jld2" vel_true vel_init conf
