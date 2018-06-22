# Scalar Helmoholtz equation solver in 2D
# Author: Li, Da
# Email: da.li1@ucalgary.ca

include("scalar_helmholtz_solver.jl");

function compute_gradient(vel, source_multi, acq_fre, fre_range, recorded_data; verbose=false)
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
        println("Computing gradient frequency range: ", frequency[fre_range]);
    end

    # Initialize
    gradient = SharedArray{Float32}(Nx*Ny, fre_num, source_num);
    recorded_forward = SharedArray{Complex64}(Nx*Ny,fre_num,source_num);
    misfit_diff = 0;
    # Extend area
    beta, vel_ex = extend_area(vel, acq_fre);
    # Source term
    # Size: Nx_pml-2 * Ny_pml-2
    source_vec = change_source(acq_fre);
    # Receiver projector
    R = build_proj_op(acq_fre.Nx,acq_fre.Ny,acq_fre.receiver_coor,acq_fre.receiver_num);

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
            r_forward_vec = R * reshape(u_forward,Nx*Ny,1);
            recorded_forward[:,ind_fre,ind_source] = r_forward_vec;
            recorded_data0 = recorded_data[:,ind_fre,ind_source];
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
            # Theorical right gradient
            # grad = real(-omega[ind_fre].^2 ./ (vel.^3) .* u_forward .* u_back);
            # Energy compensation
            # grad = real(-omega[ind_fre].^2 ./ (vel.^3) .* u_forward .* u_back) ./ (abs(u_forward)+0.1);
            # Working gradient
            grad = -real(omega[ind_fre].^2 .* u_forward .* u_back);
            gradient[:,ind_fre,ind_source] = reshape(grad, Nx*Ny, 1);
        end
        if verbose == true
            println("Frequency: ", frequency[ind_fre], " Hz complete.");
        end
    end
    gradient = sum(gradient,3);
    gradient = sum(gradient,2);
    gradient = reshape(gradient,Nx,Ny);
    return gradient
end
