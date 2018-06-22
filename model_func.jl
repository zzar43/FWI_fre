# This file contains functions for build model

function source_ricker(center_fre, center_time, t)
    x = (1 - 2*pi^2*center_fre^2*(t-center_time).^2) .* exp.(-pi^2*center_fre^2*(t-center_time).^2);
    return x;
end

function build_source_multi(center_fre,center_time,acq_fre;ricker=true)
    t = acq_fre.t;
    fre_position = acq_fre.fre_position;
    source_num = acq_fre.source_num;
    fre_num = acq_fre.fre_num;

    if ricker == true
        source_time = source_ricker(center_fre, center_time, t);
        source_fre = fft(source_time);
        source_func = source_fre[fre_position];
    else
        source_func = ones(length(fre_position));
    end

    # Outer is source index, middle is frequency index
    source_multi = zeros(Complex64,Nx*Ny,fre_num,source_num);
    for ind_source = 1:source_num
        for ind_fre = 1:fre_num
            source_mat = zeros(Complex64,Nx,Ny);
            source_mat[source_coor[ind_source,1], source_coor[ind_source,2]] = source_func[ind_fre];
            source_mat = reshape(source_mat,Nx*Ny,1);
            source_multi[:,ind_fre,ind_source] = source_mat;
        end
    end

    return source_multi
end

function build_proj_op(Nx,Ny,receiver_coor,receiver_num)
    R = spzeros(Int64,Nx*Ny,Nx*Ny);
    receiver_ind = receiver_coor[:,1] + (receiver_coor[:,2]-1)*Nx;
    for i = 1:receiver_num
        R[receiver_ind[i],receiver_ind[i]] = 1;
    end
    return R
end
#
# function build_proj_op_pml(Nx,Ny,receiver_coor,receiver_num,pml_len)
#     # This is for build the adjoint source during the adjoint method
#     Nx_pml = Nx + 2pml_len - 2;
#     Ny_pml = Ny + 2pml_len - 2;
#     R = spzeros(Int64,Nx_pml*Ny_pml,Nx_pml*Ny_pml);
#     receiver_coor += (pml_len-1);
#     receiver_ind = receiver_coor[:,1] + (receiver_coor[:,2]-1)*Nx_pml;
#     for i = 1:receiver_num
#         R[receiver_ind[i],receiver_ind[i]] = 1;
#     end
#     return R
# end

function extend_vel(vel, acq_fre)
    # return to the vector version extended velocity
    pml_len = acq_fre.pml_len;
    Nx_pml = acq_fre.Nx + 2pml_len;
    Ny_pml = acq_fre.Ny + 2pml_len;

    vel_ex = zeros(Nx_pml,Ny_pml);
    vel_ex[pml_len+1:end-pml_len,pml_len+1:end-pml_len] = vel;
    for i = 1:pml_len
        vel_ex[i,:] = vel_ex[pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
        vel_ex[:,i] = vel_ex[:,pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
    end
    vel_ex_vec = reshape(vel_ex,Nx_pml*Ny_pml,1);
    return vel_ex_vec
end
