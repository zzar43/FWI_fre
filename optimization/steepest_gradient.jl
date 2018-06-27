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

function steepest_gradient(vel_init, conf, recorded_data, vmin, vmax; alpha_1=100, alpha_max=500, iter_time=5, c1=1e-11, c2=0.9, search_time=5, zoom_time=5, verbose=false, save_graph=false, fre_range="all")

    if fre_range == "all"
        fre_range = 1:1:conf.fre_num
    end

    iter_main = 0;

    for ind_fre in fre_range
        fre_range1 = [ind_fre];
        for iter = 1:iter_time
            iter_main += 1
            if verbose == true
                println("\nSteepest Gradient Iteration: ", iter_main, " frequency: ", conf.frequency[ind_fre], " Hz.");
            end
            # Compute gradient
            grad, phi = compute_gradient(vel_init, conf, recorded_data; fre_range=fre_range1);
            # Direction
            p = -grad / maximum(grad);
            # if save_graph == true
            #     matshow((p)',cmap="RdBu", clim=[-1,1]); colorbar(); title("$iter_main");
            #     savefig("temp_graph/direction_$iter_main.png");
            #     println("Direction graph saved.")
            # end
            # Line search
            alpha = line_search(vel_init, conf, recorded_data, grad, p, phi, alpha_1, alpha_max, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)

            # update velocity
            vel_init = update_velocity(vel_init,alpha,p,vmin,vmax);

            if alpha == 0
                break;
            end
        end
        if save_graph == true
            title_name = conf.frequency[ind_fre];
            matshow((vel_init)',cmap="PuBu"); colorbar(); title("$title_name Hz");
            savefig("temp_graph/vel_$ind_fre.png");
            println("Velocity graph saved.")
        end
    end
    return vel_init
end
