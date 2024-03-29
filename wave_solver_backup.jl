# Scalar Helmoholtz equation solver in 2D
# Author: Li, Da
# Email: da.li1@ucalgary.ca
"""
    scalar_helmholtz_solver_parallel(vel, acq_fre, fre_range="all", verbose=false)

    Input:
    vel: velocity model, should in matrix form with Nx*Ny
    acq_fre: data structure of all other informations
    fre_range: frequency range need to be compute.
        For example: if frequency is [2 Hz, 3 Hz, 5 Hz], fre_range = 1:2 means 2 and 3 Hz be computed.
        If fre_range = "all", all frequency content will be computed.

"""
function scalar_helmholtz_solver(vel, source_multi, acq_fre, fre_range="all"; verbose=false)
    # This is the fundamental solver
    # Both vel and source are matrix form with size Nx*Ny
    Nx_pml = acq_fre.Nx_pml;
    Ny_pml = acq_fre.Ny_pml;
    pml_len = acq_fre.pml_len;
    Nx = acq_fre.Nx;
    Ny = acq_fre.Ny;
    h = acq_fre.h;
    frequency = acq_fre.frequency;
    omega = frequency * 2 * pi;
    fre_num = acq_fre.fre_num;
    source_num = acq_fre.source_num;

    if fre_range == "all"
        fre_range = 1:fre_num
    end
    if verbose == true
        println("Computing helmholtz equation with frequency range: ", frequency[fre_range]);
    end

    # Initialize
    wavefield = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
    recorded_data = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
    # Extend area
    beta, vel_ex = extend_area(vel, acq_fre);
    # Source term
    source_vec = change_source(source_multi, acq_fre);
    # Receiver projector
    R = build_proj_op(acq_fre.Nx,acq_fre.Ny,acq_fre.receiver_coor,acq_fre.receiver_num);

    @sync @parallel for ind_fre in fre_range
        A = make_diff_operator(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
        F = lufact(A);
        for ind_source = 1:source_num
            source = source_vec[:,ind_fre,ind_source];
            # u_vec = A\source;
            u_vec = F\source;
            u = reshape(u_vec,Nx_pml-2,Ny_pml-2);
            u = u[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
            u = reshape(u, Nx*Ny, 1);
            wavefield[:,ind_fre,ind_source] = u;
            recorded_data[:,ind_fre,ind_source] = R * u;
        end
        if verbose == true
            println("Frequency: ", frequency[ind_fre], " Hz complete.");
        end
    end
    wavefield = Array(wavefield);
    recorded_data = Array(recorded_data);
    return wavefield, recorded_data
end

# Construct the differential operator with size (Nx_pml-2)*(Ny_pml-2) by (Nx_pml-2)*(Ny_pml-2)
function make_diff_operator(h,omega,vel_ex,beta,Nx_pml,Ny_pml)
    coef = (1 + im*beta) .* (h^2*omega.^2) ./ (vel_ex.^2);
    coef = coef - 4;
    coef0 = coef[2:end-1,2:end-1];
    coef0 = reshape(coef0, (Nx_pml-2)*(Ny_pml-2));
    vec1 = ones((Nx_pml-2)*(Ny_pml-2)-Nx_pml+2);
    vec2 = ones((Nx_pml-2)*(Ny_pml-2)-1);
    B = spdiagm((vec1, vec2, coef0, vec2, vec1), (-(Nx_pml-2),-1,0,1,(Nx_pml-2)));

    for i = 1:(Ny_pml-2-1)
        ind_x = (Nx_pml-2)*i+1;
        ind_y = (Nx_pml-2)*i;
        B[ind_x,ind_y] = 0;
        B[ind_y,ind_x] = 0;
    end
    return B
end

# Extend the velocity model from Nx*Ny to Nx_pml*Ny_pml
function extend_area(vel, acq_fre)
    # Extend the velocity and build the damping factor
    pml_alpha = acq_fre.pml_alpha;
    pml_len = acq_fre.pml_len;
    Nx_pml = acq_fre.Nx_pml;
    Ny_pml = acq_fre.Ny_pml;

    pml_value = linspace(0,pml_alpha,pml_len);
    # (1+i\beta)k^2 u + \Delta u = -f.
    # beta is the damping factor
    beta = zeros(Nx_pml,Ny_pml);
    for i = 1:pml_len
        beta[pml_len+1-i,:] = pml_value[i];
        beta[end-pml_len+i,:] = pml_value[i];
        beta[:,pml_len+1-i] = pml_value[i];
        beta[:,end-pml_len+i] = pml_value[i];
    end

    vel_ex = zeros(Nx_pml,Ny_pml);
    vel_ex[pml_len+1:end-pml_len,pml_len+1:end-pml_len] = vel;
    for i = 1:pml_len
        vel_ex[i,:] = vel_ex[pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
        vel_ex[:,i] = vel_ex[:,pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
    end

    return beta, vel_ex
end

# Change the source term in the vector form [(Nx_pml-2)*(Ny_pml-2), fre_num, source_num]
function change_source(source_multi, acq_fre)
    Nx_pml = acq_fre.Nx_pml;
    Ny_pml = acq_fre.Ny_pml;
    fre_num = acq_fre.fre_num;
    source_num = acq_fre.source_num;
    pml_len = acq_fre.pml_len;

    # Source term
    source_vec = zeros(Complex64, (Nx_pml-2)*(Ny_pml-2), fre_num, source_num);
    for ind_fre = 1:fre_num
        for ind_source = 1:source_num
            source = zeros(Complex64,Nx_pml-2,Ny_pml-2);
            source[pml_len:end-pml_len+1,pml_len:end-pml_len+1] = reshape(source_multi[:,ind_fre,ind_source],acq_fre.Nx,acq_fre.Ny);
            # Here we have a -1 coefficient to correct the helmholtz equation
            source_vec[:,ind_fre,ind_source] = reshape(-1*source, (Nx_pml-2)*(Ny_pml-2), 1);
        end
    end
    return source_vec
end

function build_proj_op(Nx,Ny,receiver_coor,receiver_num)
    R = spzeros(Int64,Nx*Ny,Nx*Ny);
    receiver_ind = receiver_coor[:,1] + (receiver_coor[:,2]-1)*Nx;
    for i = 1:receiver_num
        R[receiver_ind[i],receiver_ind[i]] = 1;
    end
    return R
end

# Save the old code
# function scalar_helmholtz_solver(vel, source_multi, acq_fre, fre_range)
#     # This is the fundamental solver
#     # Both vel and source are matrix form with size Nx*Ny
#     Nx_pml = acq_fre.Nx_pml;
#     Ny_pml = acq_fre.Ny_pml;
#     pml_len = acq_fre.pml_len;
#     Nx = acq_fre.Nx;
#     Ny = acq_fre.Ny;
#     frequency = acq_fre.frequency;
#     omega = frequency * 2 * pi;
#     fre_num = acq_fre.fre_num;
#     source_num = acq_fre.source_num;
#
#     if fre_range == "all"
#         fre_range = 1:fre_num
#     end
#     # println("Computing helmholtz equation with frequency range: ", frequency[fre_range]);
#
#     # Initialize
#     wavefield = zeros(Complex64,Nx*Ny,fre_num,source_num);
#     recorded_data = zeros(Complex64,Nx*Ny,fre_num,source_num);
#     # Extend area
#     beta, vel_ex = extend_area(vel, acq_fre);
#     # Source term
#     source_vec = change_source(source_multi, acq_fre);
#
#     # print("    Frequency: ");
#     for ind_fre in fre_range
#         A = make_diff_operator(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
#         F = lufact(A);
#         for ind_source = 1:source_num
#             source = source_vec[:,ind_fre,ind_source];
#             # u_vec = A\source;
#             u_vec = F\source;
#             u = reshape(u_vec,Nx_pml-2,Ny_pml-2);
#             u = u[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
#             u = reshape(u, Nx*Ny, 1);
#             wavefield[:,ind_fre,ind_source] = u;
#             recorded_data[:,ind_fre,ind_source] = acq_fre.projection_op * u;
#         end
#         # print(frequency[ind_fre], " Hz ")
#     end
#     # println("complete.")
#
#
#     return wavefield, recorded_data
# end
# function make_diff_operator_old(h,omega,vel,beta,Nx,Ny)
#     coef = (1 + im*beta) .* (h^2*omega.^2) ./ (vel.^2);
#     coef = coef - 4;
#     # A = spzeros(Complex128, (Nx-2)*(Ny-2), (Nx-2)*(Ny-2));
#     A = spzeros(Complex64, (Nx-2)*(Ny-2), (Nx-2)*(Ny-2));
#     # A = zeros((Nx-2)*(Ny-2),(Nx-2)*(Ny-2));
#     # Center area
#     for i = 2:Nx-3
#         for j = 2:Ny-3
#             ind_row = (j-1)*(Nx-2)+i;
#             A[ind_row,ind_row] = coef[i+1,j+1];
#             A[ind_row,ind_row-1] = 1;
#             A[ind_row,ind_row+1] = 1;
#             A[ind_row,ind_row-Nx+2] = 1;
#             A[ind_row,ind_row+Nx-2] = 1;
#         end
#     end
#     # Top
#     i = 1;
#     for j = 2:Ny-3
#         ind_row = (j-1)*(Nx-2)+i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         # A[ind_row,ind_row-1] = 1;
#         A[ind_row,ind_row+1] = 1;
#         A[ind_row,ind_row-Nx+2] = 1;
#         A[ind_row,ind_row+Nx-2] = 1;
#     end
#     # Bottom
#     i = Nx-2;
#     for j = 2:Ny-3
#         ind_row = (j-1)*(Nx-2)+i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         A[ind_row,ind_row-1] = 1;
#         # A[ind_row,ind_row+1] = 1;
#         A[ind_row,ind_row-Nx+2] = 1;
#         A[ind_row,ind_row+Nx-2] = 1;
#     end
#     # Left
#     j = 1;
#     for i = 2:Nx-3
#         ind_row = (j-1)*(Nx-2)+i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         A[ind_row,ind_row-1] = 1;
#         A[ind_row,ind_row+1] = 1;
#         # A[ind_row,ind_row-Nx+2] = 1;
#         A[ind_row,ind_row+Nx-2] = 1;
#     end
#     # Right
#     j = Ny-2;
#     for i = 2:Nx-3
#         ind_row = (j-1)*(Nx-2)+i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         A[ind_row,ind_row-1] = 1;
#         A[ind_row,ind_row+1] = 1;
#         A[ind_row,ind_row-Nx+2] = 1;
#         # A[ind_row,ind_row+Nx-2] = 1;
#     end
#     # Corner
#     i = 1; j = 1;
#     ind_row = (j-1)*(Nx-2)+i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     # A[ind_row,ind_row-1] = 1;
#     A[ind_row,ind_row+1] = 1;
#     # A[ind_row,ind_row-Nx+2] = 1;
#     A[ind_row,ind_row+Nx-2] = 1;
#     i = Nx-2; j = 1;
#     ind_row = (j-1)*(Nx-2)+i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     A[ind_row,ind_row-1] = 1;
#     # A[ind_row,ind_row+1] = 1;
#     # A[ind_row,ind_row-Nx+2] = 1;
#     A[ind_row,ind_row+Nx-2] = 1;
#     i = 1; j = Ny-2;
#     ind_row = (j-1)*(Nx-2)+i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     # A[ind_row,ind_row-1] = 1;
#     A[ind_row,ind_row+1] = 1;
#     A[ind_row,ind_row-Nx+2] = 1;
#     # A[ind_row,ind_row+Nx-2] = 1;
#     i = Nx-2; j = Ny-2;
#     ind_row = (j-1)*(Nx-2)+i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     A[ind_row,ind_row-1] = 1;
#     # A[ind_row,ind_row+1] = 1;
#     A[ind_row,ind_row-Nx+2] = 1;
#     # A[ind_row,ind_row+Nx-2] = 1;
#     return A;
# end;
# This is a pml version. Just for fun.
# function acoustic_helmholtz_solver_pml(vel,Nx,Ny,omega,h,source_vec,pml_len,pml_alpha,return_vec=true)
#     Nx0 = Nx + 2pml_len;
#     Ny0 = Ny + 2pml_len;
#     # pml_alpha = pml_alpha * omega.^2 / (4*pi^2*20*maximum(abs.(vel))) * (100*h);
#     # println("pml alpha: ", pml_alpha)
#     # pml_alpha = 0.1;
#     pml_value = linspace(0,pml_alpha,pml_len);
#
#     # Extend velocity
#     vel_ex = zeros(Complex64,Nx0,Ny0);
#     for i in 1:pml_len
#         vel_ex[i,pml_len+1:pml_len+Ny] = vel[1,:];
#         vel_ex[pml_len+Nx+i , pml_len+1 : pml_len+Ny] = vel[end,:];
#         vel_ex[pml_len+1:pml_len+Nx,i] = vel[:,1];
#         vel_ex[pml_len+1 : pml_len+Nx , pml_len+Ny+i] = vel[:,end];
#     end
#     vel_ex[1:pml_len,1:pml_len] = vel[1,1];
#     vel_ex[1:pml_len,pml_len+Ny+1:end] = vel[1,end];
#     vel_ex[pml_len+Nx+1:end,1:pml_len] = vel[end,1];
#     vel_ex[pml_len+Nx+1:end,pml_len+Ny+1:end] = vel[end,end];
#     vel_ex[pml_len+1:pml_len+Nx, pml_len+1:pml_len+Ny] = vel;
#
#     # PML Coef
#     sigma_x = zeros(Complex64,Nx0,Ny0);
#     sigma_y = zeros(Complex64,Nx0,Ny0);
#     for i = 1:pml_len
#         sigma_x[pml_len+1-i,:] = pml_value[i];
#         sigma_x[pml_len+Nx+i,:] = pml_value[i];
#         sigma_y[:,pml_len+1-i] = pml_value[i];
#         sigma_y[:,pml_len+Ny+i] = pml_value[i];
#     end
#
#     # Sx = 1 ./ (ones(Complex64, Nx0, Ny0) - im .* sigma_x / omega);
#     # Sy = 1 ./ (ones(Complex64, Nx0, Ny0) - im .* sigma_y / omega);
#     Sx = 1 ./ (ones(Complex64, Nx0, Ny0) - im .* sigma_x / omega .* vel_ex);
#     Sy = 1 ./ (ones(Complex64, Nx0, Ny0) - im .* sigma_y / omega .* vel_ex);
#
#
#     coef = (omega./vel_ex).^2 - (2/h^2)*(Sx + Sy);
#     coef_x = zeros(Complex64,Nx0,Ny0);
#     coef_y = zeros(Complex64,Nx0,Ny0);
#     coef_x[2:end-1,2:end-1] = (Sx[2:end-1,2:end-1] .* (Sx[3:end,2:end-1]-Sx[1:end-2,2:end-1])) ./ (4*h.^2);
#     coef_x[1,:] = coef_x[2,:]; coef_x[end,:] = coef_x[end-1,:];
#     coef_x[:,1] = coef_x[:,2]; coef_x[:,end] = coef_x[:,end-1];
#     coef_y[2:end-1,2:end-1] = (Sy[2:end-1,2:end-1] .* (Sy[2:end-1,3:end]-Sy[2:end-1,1:end-2])) ./ (4*h.^2);
#     coef_y[1,:] = coef_y[2,:]; coef_y[end,:] = coef_y[end-1,:];
#     coef_y[:,1] = coef_y[:,2]; coef_y[:,end] = coef_y[:,end-1];
#     coef_x1 = coef_x + Sx.^2/h^2;
#     coef_x2 = -1*coef_x + Sx.^2/h^2;
#     coef_y1 = coef_y + Sy.^2/h^2;
#     coef_y2 = -1*coef_y + Sy.^2/h^2;
#
#     # build differential matrix
#     A = spzeros(Complex64,(Nx0-2)*(Ny0-2),(Nx0-2)*(Ny0-2));
#     for i = 2:Nx0-3
#         for j = 2:Ny0-3
#             ind_row = (j-1)*(Nx0-2) + i;
#             A[ind_row,ind_row] = coef[i+1,j+1];
#             A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#             A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#             A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#             A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#         end
#     end
#     # Top
#     i = 1;
#     for j = 2:Ny0-3
#         ind_row = (j-1)*(Nx0-2) + i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         # A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#         A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#         A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#         A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     end
#     # Bottom
#     i = Nx0-2;
#     for j = 2:Ny0-3
#         ind_row = (j-1)*(Nx0-2) + i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#         # A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#         A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#         A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     end
#     # Left
#     j = 1;
#     for i = 2:Nx0-3
#         ind_row = (j-1)*(Nx0-2) + i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#         A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#         # A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#         A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     end
#     # Right
#     j = Ny0-2;
#     for i = 2:Nx0-3
#         ind_row = (j-1)*(Nx0-2) + i;
#         A[ind_row,ind_row] = coef[i+1,j+1];
#         A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#         A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#         A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#         # A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     end
#     # Top Left
#     i = 1; j = 1;
#     ind_row = (j-1)*(Nx0-2) + i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     # A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#     A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#     # A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#     A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     # Top Right
#     i = 1; j = Ny0-2;
#     ind_row = (j-1)*(Nx0-2) + i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     # A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#     A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#     A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#     # A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     # Bottom Left
#     i = Nx0-2; j = 1;
#     ind_row = (j-1)*(Nx0-2) + i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#     # A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#     # A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#     A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#     # Bottom Right
#     i = Nx0-2; j = Ny0-2;
#     ind_row = (j-1)*(Nx0-2) + i;
#     A[ind_row,ind_row] = coef[i+1,j+1];
#     A[ind_row,ind_row-1] = coef_x2[i+1,j+1];
#     # A[ind_row,ind_row+1] = coef_x1[i+1,j+1];
#     A[ind_row,ind_row-Nx0+2] = coef_y2[i+1,j+1];
#     # A[ind_row,ind_row+Nx0-2] = coef_y1[i+1,j+1];
#
#     # Source Term
#     source = zeros(Complex64,Nx0-2,Ny0-2);
#     # source_coor += pml_len;
#     # source_ind = source_coor[:,1]-1 + (source_coor[:,2]-2)*(Nx0-2);
#     # source[source_ind] = -1*source_func;
#     # source_vec = reshape(source, (Nx0-2)*(Ny0-2), 1);
#     source[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny] = reshape(source_vec,Nx,Ny);
#     source_vec = reshape(source, (Nx0-2)*(Ny0-2), 1);
#
#     u_vec = A\source_vec;
#     # Iterative method
#     # u_vec = bicgstabl(A,source_vec);
#
#     u = reshape(u_vec,Nx0-2,Ny0-2);
#     u = u[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
#     if return_vec == true
#         u = reshape(u, Nx*Ny, 1);
#     end
#     # received_data = u[receiver_coor[:,1]+(receiver_coor[:,2]-1)*Nx];
#     return u
# end
