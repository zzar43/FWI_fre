include("scalar_helmholtz_solver.jl")

struct acquisition_fre
    # space and frequency
    Nx::Int64
    Ny::Int64
    h::Float32
    # time
    Nt::Int64
    dt
    t
    # frequency
    frequency::Array{Float32}
    fre_num::Int64
    # source
    source_num::Int64
    source_coor
    # receiver
    receiver_num::Int64
    receiver_coor
    projection_op
    projection_op_pml
    # PML
    pml_len::Int64
    pml_alpha::Float32
    Nx_pml::Int64
    Ny_pml::Int64
end

# Compute gradient by adjoint method.
function compute_gradient(vel, recorded_data, source_multi, acq_fre, fre_range)
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
    gradient = zeros(Float32, Nx*Ny, fre_num, source_num);
    recorded_forward = zeros(Complex64,Nx*Ny,fre_num,source_num);
    misfit_diff = 0;
    # Extend area
    beta, vel_ex = extend_area(vel, acq_fre);
    # Source term
    # Size: Nx_pml-2 * Ny_pml-2
    source_vec = change_source(source_multi, acq_fre);

    # Main loop
    print("    Frequency: ")
    for ind_fre in fre_range
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
            misfit_diff += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
        print(frequency[ind_fre], " Hz ")
    end
    gradient = sum(gradient,3);
    gradient = sum(gradient,2);
    gradient = reshape(gradient,Nx,Ny);
    println("complete.")
    return gradient, misfit_diff
end

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
            #
            # # Adjoint source
            # r_forward_vec = acq_fre.projection_op * reshape(u_forward,Nx*Ny,1);
            # recorded_forward[:,ind_fre,ind_source] = r_forward_vec;
            # source_adjoint = conj(r_forward_vec - recorded_data[:,ind_fre,ind_source]);
            # source_adjoint0 = zeros(Complex64,Nx_pml-2,Ny_pml-2);
            # source_adjoint0[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny] = reshape(source_adjoint,Nx,Ny);
            # source_adjoint = reshape(source_adjoint0, (Nx_pml-2)*(Ny_pml-2), 1);
            # source_adjoint = -source_adjoint;
            #
            # # Backward
            # u_back_vec = F\source_adjoint;
            # u_back = reshape(u_back_vec,Nx_pml-2,Ny_pml-2);
            # u_back = u_back[pml_len:pml_len-1+Nx,pml_len:pml_len-1+Ny];
            #
            # # Gradient
            # grad = real(-omega[ind_fre].^2 ./ (vel.^3) .* u_forward .* u_back);
            # gradient[:,ind_fre,ind_source] = reshape(grad, Nx*Ny, 1);

            # Misfit difference
            # misfit_diff += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
        println("Frequency: ", frequency[ind_fre], " Hz complete.")
    end
    gradient = sum(gradient,3);
    gradient = sum(gradient,2);
    gradient = reshape(gradient,Nx,Ny);
    for ind_fre = 1:fre_num
        for ind_source = 1:source_num
            misfit_diff += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
    end
    return gradient, misfit_diff
end

function backtracking_line_search(vel,p,gradient,alpha0,misfit_diff0,tau,c,iter_time,fre_range,recorded_data_true)
    if fre_range == "all"
        fre_range = 1:fre_num
    end
    println("Backtracking line search frequency range: ", frequency[fre_range]);

    m = sum(p.*gradient);
    t = -c*m;
    println(misfit_diff0," ", alpha0*t)
    if misfit_diff0 < alpha0*t
        error("c is too large");
    end
    iter = 1;
    misfit_diff_new = 0;
    alpha = alpha0;

    vel_new = vel + alpha * p;
    wavefield, recorded_forward = scalar_helmholtz_solver(vel_new, source_multi, acq_fre, fre_range);
    misfit_diff_new = 0;
    for ind_fre in fre_range
        for ind_source = 1:source_num
            misfit_diff_new += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data_true[:,ind_fre,ind_source])^2;
        end
    end
    println("Alpha: ", alpha, " iter time: ", iter);
    println("misfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);

    while (iter < iter_time) && ((misfit_diff0 - misfit_diff_new) < alpha * t)
        iter += 1;
        alpha = tau * alpha;
        vel_new = vel + alpha * p;
        wavefield, recorded_forward = scalar_helmholtz_solver(vel_new, source_multi, acq_fre, fre_range);
        misfit_diff_new = 0;
        for ind_fre in fre_range
            for ind_source = 1:source_num
                misfit_diff_new += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data_true[:,ind_fre,ind_source])^2;
            end
        end
        println("Alpha: ", alpha, " iter time: ", iter);
        println("misfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);
    end
    return alpha
end

function backtracking_line_search_parallel(vel,p,gradient,alpha0,misfit_diff0,tau,c,iter_time,fre_range,recorded_data_true)
    if fre_range == "all"
        fre_range = 1:fre_num
    end
    println("Backtracking line search frequency range: ", frequency[fre_range]);

    m = sum(p.*gradient);
    t = -c*m;
    println(misfit_diff0," ", alpha0*t)
    if misfit_diff0 < alpha0*t
        error("c is too large");
    end
    iter = 1;
    misfit_diff_new = 0;
    alpha = alpha0;

    vel_new = vel + alpha * p;
    wavefield, recorded_forward = scalar_helmholtz_solver_parallel(vel_new, source_multi, acq_fre, fre_range);
    misfit_diff_new = 0;
    for ind_fre in fre_range
        for ind_source = 1:source_num
            misfit_diff_new += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data_true[:,ind_fre,ind_source])^2;
        end
    end
    println("Alpha: ", alpha, " iter time: ", iter);
    println("misfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);

    while (iter < iter_time) && ((misfit_diff0 - misfit_diff_new) < alpha * t)
        iter += 1;
        alpha = tau * alpha;
        vel_new = vel + alpha * p;
        wavefield, recorded_forward = scalar_helmholtz_solver_parallel(vel_new, source_multi, acq_fre, fre_range);
        misfit_diff_new = 0;
        for ind_fre in fre_range
            for ind_source = 1:source_num
                misfit_diff_new += 0.5*norm(recorded_forward[:,ind_fre,ind_source]-recorded_data_true[:,ind_fre,ind_source])^2;
            end
        end
        println("Alpha: ", alpha, " iter time: ", iter);
        println("misfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);
    end
    return alpha
end
