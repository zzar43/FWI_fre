
c1 = 1e-9; c2 = 0.9;

function compute_misfit_func(vel, conf, recorded_data; fre_range="all")
    wavefield, recorded_data_new = scalar_helmholtz_solver(vel, conf; fre_range=fre_range, verbose=false)

    misfit_diff = 0;
    if fre_range == "all"
        fre_range = 1:conf.fre_num
    end
    for ind_fre in fre_range
        for ind_source = 1:conf.source_num
            misfit_diff += 0.5*norm(recorded_data_new[:,ind_fre,ind_source]-recorded_data[:,ind_fre,ind_source])^2;
        end
    end
    return misfit_diff
end

function update_velocity(vel,alpha,p,vmin,vmax)
    vel_new = vel + alpha * p;
    vel_new[vel_new.<vmin] = vmin;
    vel_new[vel_new.>vmax] = vmax;
    return vel_new
end


@time grad, phi_0 = compute_gradient(vel_init, conf, recorded_data; fre_range=[1], verbose=true);
phi_0_diff = sum((-1*grad/norm(grad)).*grad)

function zoomin(vel,p,vmin,vmax,alpha_lo,alpha_hi,phi_lo,phi_0,phi_0_diff;zoom_time::Int64=5)
    for i = 1:zoom_time
        alpha_j = 0.5*(alpha_lo+alpha_hi);
        vel_j = update_velocity(vel,alpha_j,p,vmin,vmax);
        grad_j, phi_j = compute_gradient(vel_j, conf, recorded_data; fre_range=[1]);
        if phi_j > phi_0 + c1*alpha_j*phi_0_diff || phi_j >= phi_lo
            alpha_hi = alpha_j;
            println("case 1")
        else
            phi_j_diff = sum((-1 * grad_j / norm(grad_j)) .* grad_j);
            if abs(phi_j_diff) <= -1*c2*phi_0_diff
                alpha = alpha_j;
                break
            end
            if phi_j_diff*(alpha_hi-alpha_lo) >= 0
                alpha_hi = alpha_lo;
            end
            alpha_lo = alpha_j;
            println("case 2")
        end
        println("zoom: ",i)
    end
    return alpha
end

vel = vel_init;
vmin = 0; vmax = 5500;
alpha_lo = 20;
vel_lo = update_velocity(vel,alpha_lo,p,vmin,vmax);
grad_lo, phi_lo = compute_gradient(vel_lo, conf, recorded_data; fre_range=[1], verbose=true);
phi_lo_diff = sum((-1*grad_lo/norm(grad_lo)).*grad_lo)
alpha_hi = 100;
vel_hi = update_velocity(vel,alpha_hi,p,vmin,vmax);
grad_hi, phi_hi = compute_gradient(vel_hi, conf, recorded_data; fre_range=[1], verbose=true);
phi_hi_diff = sum((-1*grad_hi/norm(grad_hi)).*grad_hi)

alpha = zoomin(vel,p,vmin,vmax,alpha_lo,alpha_hi,phi_lo,phi_0,phi_0_diff;zoom_time=5)


function ls(alpha_0, alpha_max)
    iter = 1
    phi_i_pre = phi_0;
    alpha_i_pre = alpha0;
    while iter <= 10
        alpha_i = 0.5 * (alpha_0 + alpha_max);
        vel_i = update_velocity(vel,alpha_i,p,vmin,vmax);
        grad_i, phi_i = compute_gradient(vel_i, conf, recorded_data; fre_range=[1], verbose=true);

        if (phi_i > phi_0 + c1*alpha_i*phi_0_diff) || ((phi_i >= phi_i_pre) && iter > 1)
            alpha = zoomin(alpha_i_pre,alpha_i);
            break;
        end
        grad_i_diff = sum((-1*grad_i/norm(grad_i)).*grad_i)
        if abs(grad_i_diff) <= (-1*c2*phi_0_diff)
            alpha = alpha_i;
            break;
        end
        if grad_i_diff >= 0
            alpha = zoomin(alpha_i,alpha_max)
            break;
        end
        alpha_0 = alpha_i;
        alpha_i = 0.5 * (alpha_0 + alpha_max);
        iter += 1;
    end
end



# grad = reshape(grad,conf.Nx*conf.Ny);
# p = -1 * grad ./ norm(grad);
#
# vel_new = vel_init + alpha0*p;
# matshow(vel_new');colorbar()
#
# phi0_diff = sum(p.*grad)
# @time phi_alpha0 = compute_misfit_func(vel_new, conf, recorded_data, fre_range=[1])
# @time phi0 = compute_misfit_func(vel_init, conf, recorded_data, fre_range=[1])
#
# alpha1 = - phi0_diff * alpha0^2 / (2*(phi_alpha0 - phi0 - phi0_diff*alpha0))
# vel_new = vel_new + alpha1*p;
#
# @time phi_alpha1 = compute_misfit_func(vel_new, conf, recorded_data, fre_range=[1])
# part1 = 1 / (alpha0^2*alpha1^2*(alpha1-alpha0))
# part2 = [alpha0^2 -alpha1^2; -alpha0^3 alpha1^3]
# part3 = [phi_alpha1-phi0-phi0_diff*alpha1; phi_alpha0-phi0-phi0_diff*alpha0]
# coef = part1 * part2 * part3
# a = coef[1]; b = coef[2];
# alpha2 = (-b+sqrt(b^2-3*a*phi0_diff)) / (3*a)
# # safeguard
# if alpha2 < 0.1*alpha1
#     alpha2 = 1/2*alpha1;
# end
# alpha0 = alpha1; alpha1 = alpha2;
