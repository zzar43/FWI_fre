function compute_gradient_parallel(vel, recorded_data, source_multi, acq_fre, fre_range)
    Nx_pml = acq_fre.Nx_pml;
    Ny_pml = acq_fre.Ny_pml;
    pml_len = acq_fre.pml_len;
    Nx = acq_fre.Nx;
    Ny = acq_fre.Ny;
    frequency = acq_fre.frequency;
    omega = frequency * 2 * pi;
    fre_num = acq_fre.fre_num;
    source_num = acq_fre.source_num;

    if fre_range == "all"
        fre_range = 1:fre_num
    end
    println("Computing gradient frequency range: ", frequency[fre_range]);

    # Initialize
    gradient = SharedArray{Float32}(Nx*Ny, fre_num, source_num);
    recorded_forward = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
    misfit_diff = 0;
    # Extend area
    beta, vel_ex = extend_area(vel, acq_fre);
    # Source term
    # Size: Nx_pml-2 * Ny_pml-2
    source_vec = change_source(source_multi, acq_fre);

    # Main loop
    @sync @parallel for ind_fre in fre_range
        A = make_diff_operator(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
        F = lufact(A);
        for ind_source = 1:source_num

            source = source_vec[:,ind_fre,ind_source];

            # Forward
            u_forward_vec = F\source;
            u_forward = reshape(u_forward_vec,Nx_pml-2,Ny_pml-2);
            u_forward = u_forward[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];

            # Adjoint source
            r_forward_vec = acq_fre.projection_op * reshape(u_forward,Nx*Ny,1);
            recorded_forward[:,ind_fre,ind_source] = r_forward_vec;
            source_adjoint = conj(r_forward_vec - recorded_data[:,ind_fre,ind_source]);
            source_adjoint0 = zeros(Complex64,Nx_pml-2,Ny_pml-2);
            source_adjoint0[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny] = reshape(source_adjoint,Nx,Ny);
            source_adjoint = reshape(source_adjoint0, (Nx_pml-2)*(Ny_pml-2), 1);
            source_adjoint = -source_adjoint;

            # Backward
            u_back_vec = F\source_adjoint;
            u_back = reshape(u_back_vec,Nx_pml-2,Ny_pml-2);
            u_back = u_back[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];

            # Gradient
            grad = real(-omega[ind_fre].^2 ./ (vel.^3) .* u_forward .* u_back);
            gradient[:,ind_fre,ind_source] = reshape(grad, Nx*Ny, 1);

            # Misfit difference
            # misfit_diff += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
        println("Frequency: ", frequency[ind_fre], " Hz complete.")

        # Gauss Newton
        J = spzeros(Complex64,Nx*Ny,Nx*Ny);
        vel_ex0 = vel_ex[2:end-1,2:end-1];
        for ind = 1:Nx*Ny
            c_ind = vel[ind];
            uu_vec = F\(omega[ind_fre]^2/c_ind^3 .* u_forward_vec);
            uu = reshape(uu_vec,Nx_pml-2,Ny_pml-2);
            uu = uu[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
            uu = reshape(uu, Nx*Ny, 1);
            uu = sparse(uu);
            J[:,ind] = uu;
        end
    end
    
    gradient = sum(gradient,3);
    gradient = sum(gradient,2);
    gradient = reshape(gradient,Nx,Ny);
    for ind_fre in fre_range
        for ind_source = 1:source_num
            misfit_diff += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
    end

    return gradient, misfit_diff
end
