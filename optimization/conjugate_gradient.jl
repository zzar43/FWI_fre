function conjugate_gradient(vel_init, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=1, iter_time=10, c=1e-5, tau=0.5, search_time=4, verbose=false, save_graph=false, single_fre=false)

    misfit_vec = zeros(Float32, iter_time * acq_fre.fre_num);

    if single_fre == false
        fre_range1 = 1:1:acq_fre.fre_num
    else
        fre_range1 = single_fre:single_fre
    end
    iter_main = 0;
    for ind_fre in fre_range1
        fre_range = [ind_fre];

        iter_main += 1;
        if verbose == true
            println("\nSteepest Gradient Iteration: ", iter_main, " frequency: ", acq_fre.frequency[ind_fre]);
        end
        grad = compute_gradient(vel_init, source_multi, acq_fre, fre_range, recorded_data_true, verbose=verbose);
        p0 = -grad / maximum(abs.(grad));
        s0 = p0;
        # Line search
        alpha, misfit_value = backtracking_line_search(vel_init,source_multi,acq_fre,p0,grad,recorded_data_true,vmin,vmax,alpha0,tau,c,search_time,"all",verbose=verbose);
        # update velocity
        vel_init = update_velocity(vel_init,alpha,p0,vmin,vmax);
        # record misfit function
        misfit_vec[iter_main] = misfit_value;

        for iter = 1:(iter_time-1)
            iter_main += 1
            if verbose == true
                println("\nSteepest Gradient Iteration: ", iter_main, " frequency: ", acq_fre.frequency[ind_fre], " at CG step.");
            end
            # Compute gradient
            grad = compute_gradient(vel_init, source_multi, acq_fre, fre_range, recorded_data_true, verbose);
            # Direction
            p1 = -grad / maximum(abs.(grad));
            beta_pr = sum(p1.*(p1-p0)) / sum(p0.*p0);
            beta_pr = max(0,beta_pr);
            s1 = p1 + beta_pr * s0;
            # Line search
            alpha, misfit_value = backtracking_line_search(vel_init,source_multi, acq_fre,s1,grad,recorded_data_true,vmin,vmax,alpha0,tau,c,search_time,"all",verbose=verbose);
            # update velocity
            vel_init = update_velocity(vel_init,alpha,s1,vmin,vmax);
            # record misfit function
            misfit_vec[iter_main] = misfit_value;
            s0 = s1; p0 = p1;
            if alpha == 0
                break;
            end
        end
        if save_graph == true
            title_name = acq_fre.frequency[ind_fre];
            matshow((vel_init)'); colorbar(); title("$title_name Hz")
            savefig("temp_graph/vel_$ind_fre.png");
            println("Graph saved.")
        end
    end
    misfit_vec = misfit_vec[1:iter_main];
    return vel_init, misfit_vec
end
