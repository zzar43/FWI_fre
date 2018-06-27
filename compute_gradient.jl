# Scalar Helmoholtz equation solver in 2D
# Author: Li, Da
# Email: da.li1@ucalgary.ca

include("scalar_helmholtz_solver.jl");

function compute_gradient(vel, conf, recorded_data; fre_range="all", verbose::Bool=false)
    Nx_pml = conf.Nx + 2*conf.pml_len;
    Ny_pml = conf.Ny + 2*conf.pml_len;
    omega = 2*pi*conf.frequency;

    # Change the shape of vel
    if size(vel)[2] == 1
        vel = reshape(vel,conf.Nx,conf.Ny)
    end
    if fre_range == "all"
        fre_range = 1:conf.fre_num
    end
    if verbose == true
        println("Computing gradient frequency range: ", conf.frequency[fre_range]);
    end

    # Initialize
    gradient = SharedArray{Float32}(Nx_pml*Ny_pml, conf.fre_num, conf.source_num);
    # recorded_forward = SharedArray{Complex64}(conf.receiver_num,fre_num,source_num);
    misfit_diff = 0.0;
    # Receiver projector
    R1 = build_proj_op1(conf);
    R2 = build_proj_op2(conf);

    # Main loop
    @sync @parallel for ind_fre in fre_range
        A = make_diff_operator(vel,conf,ind_fre=ind_fre);
        F = lufact(A);
        for ind_source = 1:conf.source_num
            source = make_source(conf,ind_fre=ind_fre,ind_source=ind_source);
            # Forward
            u_forward_vec = F\source; # size [Nx_pml*Ny_pml,1]
            misfit_diff += 0.5 * norm(R1*u_forward_vec - recorded_data[:,ind_fre,ind_source])^2

            # Adjoint source
            r_forward_vec = R2 * u_forward_vec; # size [Nx_pml*Ny_pml,1]
            source_adjoint = conj(r_forward_vec - R1.'*recorded_data[:,ind_fre,ind_source]);

            # Backward
            u_back_vec = F\source_adjoint; # size [Nx_pml*Ny_pml,1]

            # Gradient
            grad = real(omega[ind_fre].^2 .* u_forward_vec .* u_back_vec);
            gradient[:,ind_fre,ind_source] = grad;
        end
        if verbose == true
            println("Frequency: ", conf.frequency[ind_fre], " Hz complete.");
        end
    end
    gradient = sum(gradient,3);
    gradient = sum(gradient,2);
    gradient = reshape(gradient,Nx_pml,Ny_pml);
    gradient = gradient[conf.pml_len+1:end-conf.pml_len,conf.pml_len+1:end-conf.pml_len];
    gradient = Array(gradient);
    return gradient, misfit_diff
end

function build_proj_op2(conf)
    # projection operator 1
    Nx_pml = conf.Nx + 2*conf.pml_len;
    Ny_pml = conf.Ny + 2*conf.pml_len;
    R = spzeros(Int64,Nx_pml*Ny_pml,Nx_pml*Ny_pml);
    receiver_coor = conf.receiver_coor + conf.pml_len;

    receiver_ind = receiver_coor[:,1] + (receiver_coor[:,2]-1)*Nx_pml;
    for i = 1:conf.receiver_num
        R[receiver_ind[i],receiver_ind[i]] = 1;
    end
    return R
end

#
# function compute_gradient(vel, source_multi, acq_fre, fre_range, recorded_data; verbose=false)
#     Nx_pml = acq_fre.Nx_pml;
#     Ny_pml = acq_fre.Ny_pml;
#     pml_len = acq_fre.pml_len;
#     Nx = acq_fre.Nx;
#     Ny = acq_fre.Ny;
#     h = acq_fre.h;
#     frequency = acq_fre.frequency;
#     omega = frequency * 2 * pi;
#     fre_num = acq_fre.fre_num;
#     source_num = acq_fre.source_num;
#
#     if fre_range == "all"
#         fre_range = 1:fre_num
#     end
#     if verbose == true
#         println("Computing gradient frequency range: ", frequency[fre_range]);
#     end
#
#     # Initialize
#     gradient = SharedArray{Float32}(Nx*Ny, fre_num, source_num);
#     recorded_forward = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
#     misfit_diff = 0;
#     # Extend area
#     beta, vel_ex = extend_area(vel, acq_fre);
#     # Source term
#     # Size: Nx_pml-2 * Ny_pml-2
#     source_vec = change_source(source_multi,acq_fre);
#     # Receiver projector
#     R = build_proj_op(acq_fre.Nx,acq_fre.Ny,acq_fre.receiver_coor,acq_fre.receiver_num);
#
#     # Main loop
#     @sync @parallel for ind_fre in fre_range
#         A = make_diff_operator(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
#         F = lufact(A);
#         for ind_source = 1:source_num
#             source = source_vec[:,ind_fre,ind_source];
#
#             # Forward
#             u_forward_vec = F\source;
#             u_forward = reshape(u_forward_vec,Nx_pml-2,Ny_pml-2);
#             u_forward = u_forward[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
#
#             # Adjoint source
#             r_forward_vec = R * reshape(u_forward,Nx*Ny,1);
#             recorded_forward[:,ind_fre,ind_source] = r_forward_vec;
#             recorded_data0 = recorded_data[:,ind_fre,ind_source];
#             source_adjoint = conj(r_forward_vec - recorded_data[:,ind_fre,ind_source]);
#             source_adjoint0 = zeros(Complex64,Nx_pml-2,Ny_pml-2);
#             source_adjoint0[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny] = reshape(source_adjoint,Nx,Ny);
#             source_adjoint = reshape(source_adjoint0, (Nx_pml-2)*(Ny_pml-2), 1);
#             source_adjoint = -source_adjoint;
#
#             # Backward
#             u_back_vec = F\source_adjoint;
#             u_back = reshape(u_back_vec,Nx_pml-2,Ny_pml-2);
#             u_back = u_back[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
#
#             # Gradient
#             # Theorical right gradient
#             # grad = real(-omega[ind_fre].^2 ./ (vel.^3) .* u_forward .* u_back);
#             # Energy compensation
#             # grad = real(-omega[ind_fre].^2 ./ (vel.^3) .* u_forward .* u_back) ./ (abs(u_forward)+0.1);
#             # Working gradient
#             grad = -real(omega[ind_fre].^2 .* u_forward .* u_back);
#             gradient[:,ind_fre,ind_source] = reshape(grad, Nx*Ny, 1);
#         end
#         if verbose == true
#             println("Frequency: ", frequency[ind_fre], " Hz complete.");
#         end
#     end
#     gradient = sum(gradient,3);
#     gradient = sum(gradient,2);
#     gradient = reshape(gradient,Nx,Ny);
#     return gradient
# end
# # Extend the velocity model from Nx*Ny to Nx_pml*Ny_pml
# function extend_area(vel, acq_fre)
#     # Extend the velocity and build the damping factor
#     pml_alpha = acq_fre.pml_alpha;
#     pml_len = acq_fre.pml_len;
#     Nx_pml = acq_fre.Nx_pml;
#     Ny_pml = acq_fre.Ny_pml;
#
#     pml_value = linspace(0,pml_alpha,pml_len);
#     # (1+i\beta)k^2 u + \Delta u = -f.
#     # beta is the damping factor
#     beta = zeros(Nx_pml,Ny_pml);
#     for i = 1:pml_len
#         beta[pml_len+1-i,:] = pml_value[i];
#         beta[end-pml_len+i,:] = pml_value[i];
#         beta[:,pml_len+1-i] = pml_value[i];
#         beta[:,end-pml_len+i] = pml_value[i];
#     end
#
#     vel_ex = zeros(Nx_pml,Ny_pml);
#     vel_ex[pml_len+1:end-pml_len,pml_len+1:end-pml_len] = vel;
#     for i = 1:pml_len
#         vel_ex[i,:] = vel_ex[pml_len+1,:];
#         vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
#         vel_ex[:,i] = vel_ex[:,pml_len+1];
#         vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
#     end
#
#     return beta, vel_ex
# end
# # Change the source term in the vector form [(Nx_pml-2)*(Ny_pml-2), fre_num, source_num]
# function change_source(source_multi, acq_fre)
#     Nx_pml = acq_fre.Nx_pml;
#     Ny_pml = acq_fre.Ny_pml;
#     fre_num = acq_fre.fre_num;
#     source_num = acq_fre.source_num;
#     pml_len = acq_fre.pml_len;
#
#     # Source term
#     source_vec = zeros(Complex64, (Nx_pml-2)*(Ny_pml-2), fre_num, source_num);
#     for ind_fre = 1:fre_num
#         for ind_source = 1:source_num
#             source = zeros(Complex64,Nx_pml-2,Ny_pml-2);
#             source[pml_len:end-pml_len+1,pml_len:end-pml_len+1] = reshape(source_multi[:,ind_fre,ind_source],acq_fre.Nx,acq_fre.Ny);
#             # Here we have a -1 coefficient to correct the helmholtz equation
#             source_vec[:,ind_fre,ind_source] = reshape(-1*source, (Nx_pml-2)*(Ny_pml-2), 1);
#         end
#     end
#     return source_vec
# end
#
# function build_proj_op(Nx,Ny,receiver_coor,receiver_num)
#     R = spzeros(Int64,Nx*Ny,Nx*Ny);
#     receiver_ind = receiver_coor[:,1] + (receiver_coor[:,2]-1)*Nx;
#     for i = 1:receiver_num
#         R[receiver_ind[i],receiver_ind[i]] = 1;
#     end
#     return R
# end
# # Construct the differential operator with size (Nx_pml-2)*(Ny_pml-2) by (Nx_pml-2)*(Ny_pml-2)
# function make_diff_operator(h,omega,vel_ex,beta,Nx_pml,Ny_pml)
#     coef = (1 + im*beta) .* (h^2*omega.^2) ./ (vel_ex.^2);
#     coef = coef - 4;
#     coef0 = coef[2:end-1,2:end-1];
#     coef0 = reshape(coef0, (Nx_pml-2)*(Ny_pml-2));
#     vec1 = ones((Nx_pml-2)*(Ny_pml-2)-Nx_pml+2);
#     vec2 = ones((Nx_pml-2)*(Ny_pml-2)-1);
#     B = spdiagm((vec1, vec2, coef0, vec2, vec1), (-(Nx_pml-2),-1,0,1,(Nx_pml-2)));
#
#     for i = 1:(Ny_pml-2-1)
#         ind_x = (Nx_pml-2)*i+1;
#         ind_y = (Nx_pml-2)*i;
#         B[ind_x,ind_y] = 0;
#         B[ind_y,ind_x] = 0;
#     end
#     return B
# end
