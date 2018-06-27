function zoomin(alpha_hi, alpha_lo, phi_lo, phi_0, phi_0_diff, vel_init, p, c1, c2, vmin, vmax, fre_range, zoom_time)
    alpha = 0;
    println("Start zoom.")

    for j = 1:zoom_time
        alpha_j = 0.5 * (alpha_hi + alpha_lo);
        println("Zoom time: ", j, " alpha_lo: ", alpha_lo, " alpha_j: ", alpha_j, " alpha_hi: ",alpha_hi)
        vel_j = update_velocity(vel_init,alpha_j,p,vmin,vmax);
        grad_j, phi_j = compute_gradient(vel_j, conf, recorded_data; fre_range=fre_range, verbose=false);
        println("phi_j: ", phi_j, " phi_0 + c1*alpha_j*phi_0_diff: ", phi_0 + c1*alpha_j*phi_0_diff, " phi_lo: ", phi_lo)

        if (phi_j>(phi_0 + c1*alpha_j*phi_0_diff)) || (phi_j >= phi_lo)
            alpha_hi = alpha_j;
            # println("alpha_hi = alpha_j")
        else
            # phi_j_diff = sum((-1*grad_j./maximum(grad_j)) .* grad_j);
            phi_j_diff = sum(p .* grad_j);
            println("phi_j_diff: ", phi_j_diff, " -c2 * phi_0_diff: ", -c2 * phi_0_diff);
            if (abs(phi_j_diff) <= -c2 * phi_0_diff)
                # println("alpha is: ", alpha_j)
                alpha = alpha_j;
                break;
            end
            if phi_j_diff*(alpha_hi-alpha_lo) >= 0
                println("phi_j_diff*(alpha_hi-alpha_lo): ", phi_j_diff*(alpha_hi-alpha_lo), " hi and lo upsidedown.")
                alpha_hi = alpha_lo;
            end
            alpha_lo = alpha_j;
        end

        # update
        phi_pre = phi_j;
    end

    if alpha == 0
        println("Zoom fail, alpha = alpha_j: ", alpha);
    else
        println("Zoom succeed, alpha is: ", alpha)
    end
    return alpha
end

function line_search(vel_init, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, fre_range, vmin, vmax; c1 = 1e-11, c2=0.9, search_time=5, zoom_time=5)
    phi_0_diff = sum(p_0 .* grad_0);
    println("\nStart line search.\nCheck the coefficients:")
    println("phi_0: ", phi_0, " phi_0 + c1*alpha_1*phi_0_diff: ", phi_0 + c1*alpha_1*phi_0_diff)

    iter = 1;
    alpha = 0;
    alpha_0 = 0;
    while iter <= search_time
        println("Search time: ", iter, " alpha_1: ", alpha_1)
        vel_1 = update_velocity(vel_init,alpha_1,p_0,vmin,vmax);
        grad_1, phi_1 = compute_gradient(vel_1, conf, recorded_data; fre_range=fre_range, verbose=false);
        println("phi_1: ", phi_1, "; (phi_0 + c1*alpha_1*phi_0_diff): ", phi_0 + c1*alpha_1*phi_0_diff, " phi_0: ",phi_0);
        if (phi_1>(phi_0 + c1*alpha_1*phi_0_diff)) || ((phi_1>=phi_0)&&(iter>1))
            alpha = zoomin(alpha_1, alpha_0, phi_0, phi_0, phi_0_diff, vel_init, p_0, c1, c2, vmin, vmax, fre_range, zoom_time)
            break;
        end
        # phi_1_diff = sum((-grad_1./maximum(grad_1)) .* grad_1)
        phi_1_diff = sum(p_0 .* grad_1)
        println("phi_1_diff: ", phi_1_diff, " -c2*phi_0_diff: ", -c2*phi_0_diff);
        if abs(phi_1_diff) <= -c2*phi_0_diff
            alpha = alpha_1;
            break;
        end
        if phi_1_diff >= 0
            alpha = zoomin(alpha_0, alpha_1, phi_1, phi_0, phi_0_diff, vel_init, p_0, c1, c2, vmin, vmax, fre_range, zoom_time);
            break;
        end
        # update
        alpha_0 = alpha_1
        alpha_1 = 1.5*alpha_1;
        iter += 1
        if alpha >= alpha_max
            break;
        end
    end
    if alpha == 0
        println("Search fail, alpha is: ", alpha)
    else
        println("Search succeed, alpha is: ", alpha)
    end
    return alpha
end

function backtracking_line_search(vel,source_multi,acq_fre,p,gradient,recorded_data,vmin,vmax,alpha,tau,c,search_time,fre_range;verbose=true)
    if fre_range == "all"
        fre_range = 1:acq_fre.fre_num
    end
    if verbose == true
        println("Backtracking line search frequency range: ", acq_fre.frequency[fre_range]);
    end

    m = sum(p.*gradient);
    t = -c*m;
    iter = 1;

    vel_new = update_velocity(vel,alpha,p,vmin,vmax);
    misfit_diff0 = compute_misfit_func(vel, source_multi, acq_fre, recorded_data, fre_range);
    misfit_diff_new = compute_misfit_func(vel_new, source_multi, acq_fre, recorded_data, fre_range);

    if verbose == true
        println("Alpha: ", alpha, " search time: ", iter, "\nmisfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);
    end
    while (iter < search_time) && ((misfit_diff0 - misfit_diff_new) < alpha * t)
        alpha = tau * alpha;
        vel_new = update_velocity(vel,alpha,p,vmin,vmax);
        misfit_diff_new = compute_misfit_func(vel_new, source_multi, acq_fre, recorded_data, fre_range);
        iter += 1;
        if verbose == true
            println("Alpha: ", alpha, " search time: ", iter, "\nmisfit_diff0: ", misfit_diff0, " misfit_diff_new: ", misfit_diff_new, " difference: ", misfit_diff0-misfit_diff_new, " αt: ", alpha * t);
        end
    end
    # if misfit_diff0 < misfit_diff_new
    if (misfit_diff0 - misfit_diff_new) < alpha * t
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

function compute_misfit_func(vel, source_multi, acq_fre, recorded_data, fre_range)
    wavefield, recorded_data_new = scalar_helmholtz_solver(vel, source_multi, acq_fre, fre_range, verbose=false);
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
    # vel_new[find(x->(x<vmin),vel_new)] = vmin;
    # vel_new[find(x->(x>vmax),vel_new)] = vmax;
    vel_new[vel_new.<vmin] = vmin;
    vel_new[vel_new.>vmax] = vmax;
    return vel_new
end
