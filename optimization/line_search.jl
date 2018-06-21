function backtracking_line_search(vel,acq_fre,p,gradient,recorded_data,vmin,vmax,alpha,tau,c,search_time,fre_range,verbose=true)
    if fre_range == "all"
        fre_range = 1:acq_fre.fre_num
    end
    if verbose == true
        println("Backtracking line search frequency range: ",   acq_fre.frequency[fre_range]);
    end

    m = sum(p.*gradient);
    t = -c*m;
    iter = 1;

    vel_new = update_velocity(vel,alpha,p,vmin,vmax);
    misfit_diff0 = compute_misfit_func(vel, acq_fre, recorded_data, fre_range);
    misfit_diff_new = compute_misfit_func(vel_new, acq_fre, recorded_data, fre_range);

    if verbose == true
        println("Alpha: ", alpha, " search time: ", iter, "\nmisfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);
    end
    while (iter < search_time) && ((misfit_diff0 - misfit_diff_new) < alpha * t)
        alpha = tau * alpha;
        vel_new = update_velocity(vel,alpha,p,vmin,vmax);
        misfit_diff_new = compute_misfit_func(vel_new, acq_fre, recorded_data, fre_range);
        iter += 1;
        if verbose == true
            println("Alpha: ", alpha, " search time: ", iter, "\nmisfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);
        end
    end
    if misfit_diff0 < misfit_diff_new
        # Two ways
        alpha = 0;
        # alpha = tau * alpha;
        misfit_diff_new = misfit_diff0;
    end
    if verbose == true
        println("Alpha is ", alpha, " misfit functional: ", misfit_diff_new)
    end
    return alpha, misfit_diff_new
end

function compute_misfit_func(vel, acq_fre, recorded_data, fre_range)
    wavefield, recorded_data_new = scalar_helmholtz_solver_parallel(vel, acq_fre, fre_range, false);
    misfit_diff = 0;
    if fre_range == "all"
        fre_range = 1:acq_fre.fre_num
    end
    for ind_fre in fre_range
        for ind_source = 1:acq_fre.source_num
            misfit_diff += 0.5*norm(recorded_data_new[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
    end
    return misfit_diff
end

function update_velocity(vel,alpha,p,vmin,vmax)
    vel_new = vel + alpha * p;
    vel_new[find(x->(x<vmin),vel_new)] = vmin;
    vel_new[find(x->(x>vmax),vel_new)] = vmax;
    return vel_new
end
