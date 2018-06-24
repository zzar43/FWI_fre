# Scalar Helmoholtz equation solver in 2D
# Author: Li, Da
# Email: da.li1@ucalgary.ca
"""
    This function is the steepest gradient method for FWI. The optimization works for each frequency from low to high.

    steepest_gradient(vel_init, acq_fre, recorded_data_true, vmin, vmax; alpha0=1, iter_time=10, c=1e-5, tau=0.5, search_time=4, verbose=false, save_graph=false)

    Input:
    vel_init: initial velocity model
    acq_fre: data structure of all other informations
    recorded_data_true: observation data
    vmin, vmax: velocity bound
    alpha0: first step size, should be about 0.1*maximum(vel_true)
    iter_time: iteration time at each frequency
    search_time: line search time

"""

function steepest_gradient(vel_init, source_multi, acq_fre, recorded_data_true, vmin, vmax; alpha0=1, iter_time=10, c=1e-5, tau=0.5, search_time=4, verbose=false, save_graph=false, save_misfit=false, single_fre=false)

    misfit_vec = zeros(Float32, iter_time * acq_fre.fre_num);

    if single_fre == false
        fre_range1 = 1:1:acq_fre.fre_num
    else
        fre_range1 = single_fre:single_fre
    end
    iter_main = 0;
    for ind_fre in fre_range1
        fre_range = [ind_fre];
        for iter = 1:iter_time
            iter_main += 1
            if verbose == true
                println("\nSteepest Gradient Iteration: ", iter_main, " frequency: ", acq_fre.frequency[ind_fre]);
            end
            # Compute gradient
            grad = compute_gradient(vel_init, source_multi, acq_fre, fre_range,  recorded_data_true, verbose=verbose);
            # Direction
            p = -grad / maximum(abs.(grad));
            if save_graph == true
                matshow((p)',cmap="RdBu", clim=[-1,1]); colorbar(); title("$iter_main");
                savefig("temp_graph/direction_$iter_main.png");
                println("Direction graph saved.")
            end
            # Line search
            alpha, misfit_value = backtracking_line_search(vel_init,source_multi,acq_fre,p,grad,recorded_data_true,vmin,vmax,alpha0,tau,c,search_time,fre_range,verbose=verbose);
            # update velocity
            vel_init = update_velocity(vel_init,alpha,p,vmin,vmax);

            if alpha == 0
                break;
            else
                # record misfit function
                if save_misfit == true
                    misfit_value = compute_misfit_func(vel_init, source_multi, acq_fre, recorded_data_true, "all");
                    println("Misfit value is: ", misfit_value)
                else
                    misfit_value = 0;
                end
                misfit_vec[iter_main] = misfit_value;
            end
        end
        if save_graph == true
            title_name = acq_fre.frequency[ind_fre];
            matshow((vel_init)',cmap="PuBu"); colorbar(); title("$title_name Hz");
            savefig("temp_graph/vel_$ind_fre.png");
            println("Velocity graph saved.")
        end
    end
    misfit_vec = misfit_vec[1:iter_main];
    return vel_init, misfit_vec
end
