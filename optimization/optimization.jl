

function steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax; alpha0=1, iter_time=10, c=1e-5, tau=0.5, search_time=4, mode="all", verbose=false, save_graph=false)


    if mode == "all"
        iter_time_new = iter_time;
    elseif mode == "each_frequency"
        iter_time_new = acq_fre.fre_num * iter_time;
    else
        error("Check the mode.")
    end

    misfit_vec = zeros(iter_time_new);

    for iter_main = 1:iter_time_new
        if mode == "all"
            fre_range = "all"
        elseif mode == "each_frequency"
            fre_range = [floor(Int64,(iter_main-1)/iter_time+1)];
        end
        if verbose == true
            println("\nSteepest Gradient Iteration: ", iter_main)
        end
        # Compute gradient
        grad = compute_gradient_parallel(vel_init, acq_fre, fre_range,  recorded_data_true, verbose);
        # p = -grad / norm(grad);
        p = -grad / maximum(abs.(grad));
        alpha, misfit_value = backtracking_line_search(vel_init,acq_fre,p,grad,recorded_data_true,vmin,vmax,alpha0,tau,c,search_time,"all",verbose);
        # if alpha == 0
        #     break;
        # end
        # update velocity
        vel_init = update_velocity(vel_init,alpha,p,vmin,vmax);
        # record misfit function
        misfit_vec[iter_main] = misfit_value;

        # save_graph
        if save_graph == true
            if mode == "all"
                matshow((vel_init)'); colorbar();savefig("temp_graph/vel_$iter_main.png");
            elseif mode == "each_frequency"
                if rem((iter_main+1), iter_time) == 1
                    ind = floor(Int64,(iter_main-1)/iter_time+1);
                    matshow((vel_init)'); colorbar();savefig("temp_graph/vel_$ind.png");
                end
            end
        end
    end
    return vel_init, misfit_vec
end
