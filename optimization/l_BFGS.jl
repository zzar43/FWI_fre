function l_BFGS(vel_init, conf, recorded_data, vmin, vmax; m=3, alpha_1=100, alpha_max=500, iter_time=5, c1=1e-11, c2=0.9, search_time=5, zoom_time=5, verbose=false, save_graph=false, fre_range="all")
    if iter_time <= (m+1)
        error("iter_time should be larger than m+1")
    end
    # Initialize
    iter = 1;
    vel_init = reshape(vel_init, conf.Nx*conf.Ny, 1);

    for iter_fre = 1:conf.fre_num

        fre_range1 = [ind_fre];
        S = zeros(conf.Nx*conf.Ny,m);
        Y = zeros(conf.Nx*conf.Ny,m);

        # First
        println("Iteration: ", iter, ", before l-BFGS.")
        grad_0, phi_0 = compute_gradient(vel_init, conf, recorded_data; fre_range=fre_range1);
        p_0 = -grad_0 / maximum(grad_0);
        alpha = line_search(vel_init, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)
        vel_0 = update_velocity(vel_init,alpha,p_0,vmin,vmax);

        for iter = 2:(m+1)
            iter += 1;

            println("Iteration: ", iter, ", before l-BFGS.")
            grad_1, phi_1 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range1);
            p_1 = -grad_1 / maximum(grad_1);
            alpha = line_search(vel_init, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, fre_range, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)

            if alpha == 0
                println("Alpha is 0. Try to increase search time or alpha_1.")
                break;
            end

            vel_1 = update_velocity(vel_0,alpha,p_1,vmin,vmax);
            # Coef
            s = vel_1 - vel_0;
            y = grad_1 - grad_0;
            S[:,iter-1] = s; Y[:,iter-1] = y;
            # update
            vel_0[:] = vel_1; grad_1[:] = grad_0;
        end

        # l-BFGS
        for iter = m+2:1:iter_time
            iter += 1;
            println("Iteration: ", iter, ", l-BFGS.")

            grad_1, phi_1 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range1);
            q = grad_1;
            alpha_i_save = zeros(m)
            for i = m:-1:1
                rho_i = 1 ./ (Y[:,i].' * S[:,i]); rho_i = rho_i[1];
                alpha_i = rho_i * S[:,i].' * q; alpha_i = alpha_i[1];
                q = q - alpha_i * Y[:,i]
                alpha_i_save[i] = alpha_i;
            end
            # Descent direction
            p_1 = -r;
            # line search
            alpha = line_search(vel_0, conf, recorded_data, grad_1, p_1, phi_1, 1, 2, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time);
            if alpha == 0
                println("Alpha is 0. Break.")
                break;
            end
            # update velocity
            vel_1 = update_velocity(vel_0,alpha,p_1,vmin,vmax);
            # update coef
            s = vel_1 - vel_0
            y = grad_1 - grad_0
            S[:,1:(m-1)] = S[:,2:m]
            Y[:,1:(m-1)] = Y[:,2:m]
            S[:,3] = s; Y[:,3] = y;
            vel_0[:] = vel_1; grad_1[:] = grad_0;
        end
        if save_graph == true
           title_name = conf.frequency[ind_fre];
           matshow((vel_init)',cmap="PuBu"); colorbar(); title("$title_name Hz")
           savefig("temp_graph/vel_$ind_fre.png");
           println("Velocity graph saved.")
       end
    end
    return vel_1;
end
