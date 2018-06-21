
function steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax; alpha0=1, iter_time=10, c=1e-5, tau=0.5, search_time=4, fre_range="all", verbose=false, save_graph=false)

    if fre_range == "all"
        for i = 1:(iter_time-1)
            fre_range = [fre_range "all"];
        end
    elseif length(fre_range) < iter_time
        error("Check the fre_range.")
    end

    misfit_vec = zeros(iter_time);

    for iter_main = 1:iter_time
        if verbose == true
            println("\nSteepest Gradient Iteration: ", iter_main)
        end
        # Compute gradient
        grad = compute_gradient_parallel(vel_init, acq_fre, fre_range[iter_main], recorded_data_true, verbose);
        p = -grad / norm(grad);
        alpha, misfit_value = backtracking_line_search(vel_init,acq_fre,p,grad,recorded_data_true,vmin,vmax,alpha0,tau,c,search_time,"all",verbose);
        # update velocity
        vel_init = update_velocity(vel_init,alpha,p,vmin,vmax);
        # record misfit function
        misfit_vec[iter_main] = misfit_value;
        # save_graph
        if save_graph == true
            matshow((vel_init)'); colorbar();savefig("temp_graph/vel_$iter_main.png");
        end
    end
    return vel_init, misfit_vec
end
