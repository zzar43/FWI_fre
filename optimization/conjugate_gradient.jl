function conjugate_gradient(vel_init, conf, recorded_data, vmin, vmax; alpha_1=100, alpha_max=500, iter_time=5, c1=1e-11, c2=0.9, search_time=5, zoom_time=5, verbose=false, save_graph=false, fre_range="all")

    if fre_range == "all"
        fre_range = 1:1:conf.fre_num
    end
    iter_main = 0;

    for ind_fre in fre_range

        fre_range1 = [ind_fre];
        iter_main += 1;

        if verbose == true
            println("\nConjugate Gradient Iteration: ", iter_main, " frequency: ", conf.frequency[ind_fre], " Hz.");
        end

        grad, phi = compute_gradient(vel_init, conf, recorded_data; fre_range=fre_range1);
        p0 = -grad / maximum(grad);
        s0 = p0;
        # Line search
        alpha = line_search(vel_init, conf, recorded_data, grad, s0, phi, alpha_1, alpha_max, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)
        # update velocity
        vel_init = update_velocity(vel_init,alpha,s0,vmin,vmax);

        # if save_graph == true
        #     matshow((s0)',cmap="RdBu", clim=[-1,1]); colorbar(); title("$iter_main");
        #     savefig("temp_graph/direction_$iter_main.png");
        #     println("Direction graph saved.")
        # end

        for iter = 1:(iter_time-1)
            iter_main += 1
            if verbose == true
                println("\nConjugate Gradient Iteration: ", iter_main, " frequency: ", conf.frequency[ind_fre], " Hz at CG step.");
            end
            # Compute gradient
            grad, phi = compute_gradient(vel_init, conf, recorded_data; fre_range=fre_range1);
            # Direction
            p1 = -grad / maximum(grad);
            beta_pr = sum(p1.*(p1-p0)) / sum(p0.*p0);
            beta_pr = max(0,beta_pr);
            s1 = p1 + beta_pr * s0;
            # Line search
            alpha = line_search(vel_init, conf, recorded_data, grad, s1, phi, alpha_1, alpha_max, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)
            # update velocity
            vel_init = update_velocity(vel_init,alpha,s1,vmin,vmax);
            # if save_graph == true
            #     matshow((s1)',cmap="RdBu", clim=[-1,1]); colorbar(); title("$iter_main");
            #     savefig("temp_graph/direction_$iter_main.png");
            #     println("Direction graph saved.")
            # end
            s0 = s1; p0 = p1;
            if alpha == 0
                break;
            end
        end
        if save_graph == true
            title_name = conf.frequency[ind_fre];
            matshow((vel_init)',cmap="PuBu"); colorbar(); title("$title_name Hz")
            savefig("temp_graph/vel_$ind_fre.png");
            println("Velocity graph saved.")
        end
    end
    return vel_init
end
