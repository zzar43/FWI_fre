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

function scalar_helmholtz_solver(vel, conf; fre_range="all", verbose::Bool=false)
    # This is the fundamental solver
    Nx_pml = conf.Nx + 2*conf.pml_len;
    Ny_pml = conf.Ny + 2*conf.pml_len;

    # Change the shape of vel
    if size(vel)[2] == 1
        vel = reshape(vel,conf.Nx,conf.Ny)
    end
    if fre_range == "all"
        fre_range = 1:conf.fre_num
    end
    if verbose == true
        println("Computing helmholtz equation with frequency range: ", conf.frequency[fre_range]);
    end

    # Initialize
    wavefield = SharedArray{Complex64}(conf.Nx,conf.Ny,conf.fre_num,conf.source_num);
    recorded_data = SharedArray{Complex64}(conf.receiver_num,conf.fre_num,conf.source_num);
    # Receiver projector
    R = build_proj_op1(conf);

    @sync @parallel for ind_fre in fre_range
        A = make_diff_operator(vel,conf,ind_fre=ind_fre);
        F = lufact(A);
        for ind_source = 1:conf.source_num
            source = make_source(conf,ind_fre=ind_fre,ind_source=ind_source);
            u_vec = F\source;

            recorded_data[:,ind_fre,ind_source] = R * u_vec;

            u = reshape(u_vec,Nx_pml,Ny_pml);
            u = u[conf.pml_len+1:end-conf.pml_len,conf.pml_len+1:end-conf.pml_len];
            wavefield[:,:,ind_fre,ind_source] = u;
        end
        if verbose == true
            println("Frequency: ", conf.frequency[ind_fre], " Hz complete.");
        end
    end

    if (conf.source_num == 1) && (conf.fre_num == 1)
        wavefield = wavefield[:,:,1,1];
        recorded_data = recorded_data[:,1,1];
    end

    wavefield = Array(wavefield);
    recorded_data = Array(recorded_data);
    return wavefield, recorded_data
end

function make_diff_operator(vel,conf;ind_fre::Int64=1)
    Nx_pml = conf.Nx + 2*conf.pml_len;
    Ny_pml = conf.Ny + 2*conf.pml_len;
    pml_value = linspace(0,conf.pml_alpha,conf.pml_len);
    omega = 2*pi*conf.frequency[ind_fre];

    beta = zeros(Complex64,Nx_pml,Ny_pml);
    for i = 1:conf.pml_len
        beta[conf.pml_len+1-i,:] = pml_value[i];
        beta[end-conf.pml_len+i,:] = pml_value[i];
        beta[:,conf.pml_len+1-i] = pml_value[i];
        beta[:,end-conf.pml_len+i] = pml_value[i];
    end
    beta = reshape(beta,Nx_pml*Ny_pml);

    vel_ex = zeros(Complex64,Nx_pml,Ny_pml);
    vel_ex[conf.pml_len+1:end-conf.pml_len,conf.pml_len+1:end-conf.pml_len] = vel;
    for i = 1:conf.pml_len
        vel_ex[i,:] = vel_ex[conf.pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-conf.pml_len,:];
        vel_ex[:,i] = vel_ex[:,conf.pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-conf.pml_len];
    end
    vel_ex = reshape(vel_ex,Nx_pml*Ny_pml);

    coef = (1 + im*beta) .* (conf.h^2*omega.^2) ./ (vel_ex.^2) - 4;
    vec1 = ones(Nx_pml*Ny_pml-Nx_pml);
    vec2 = ones(Nx_pml*Ny_pml-1);
    A = spdiagm((vec1,vec2,coef,vec2,vec1),(-Nx_pml,-1,0,1,Nx_pml));
    for i = 1:(Ny_pml-1)
        ind_x = Nx_pml*i+1;
        ind_y = Nx_pml*i;
        A[ind_x,ind_y] = 0;
        A[ind_y,ind_x] = 0;
    end
    A = A ./ conf.h^2;

    return A
end

function make_source(conf;ind_fre::Int64=1,ind_source::Int64=1)
    # Change the coordinates into pml version
    source_coor = conf.source_coor[ind_source,:] + conf.pml_len;
    Nx_pml = conf.Nx + 2*conf.pml_len;
    Ny_pml = conf.Ny + 2*conf.pml_len;

    source_vec = zeros(Complex64,Nx_pml*Ny_pml,1);
    source_vec[source_coor[1]+(source_coor[2]-1)*Nx_pml,1] = -1*conf.source_value[ind_fre];
    return source_vec
end

function build_proj_op1(conf)
    # projection operator 1
    Nx_pml = conf.Nx + 2*conf.pml_len;
    Ny_pml = conf.Ny + 2*conf.pml_len;
    R = spzeros(Int64,conf.receiver_num,Nx_pml*Ny_pml);
    receiver_coor = conf.receiver_coor + conf.pml_len;

    receiver_ind = receiver_coor[:,1] + (receiver_coor[:,2]-1)*Nx_pml;
    for i = 1:conf.receiver_num
        R[i,receiver_ind[i]] = 1;
    end
    return R
end
