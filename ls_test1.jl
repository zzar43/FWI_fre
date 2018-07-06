# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");
@everywhere include("forward_modelling.jl");
# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" wavefield_true recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);

function zoomin(alpha_hi, alpha_lo, phi_lo, vel_init, p, vmin, vmax, fre_range, zoom_time)
    alpha = 0;
    println("\nStart zoom.")
    for j = 1:zoom_time
        alpha_j = 0.5 * (alpha_hi + alpha_lo);
        println("Zoom time: ", j, " alpha_lo: ", alpha_lo, " alpha_j: ", alpha_j, " alpha_hi: ",alpha_hi)
        vel_j = update_velocity(vel_init,alpha_j,p,vmin,vmax);
        grad_j, phi_j = compute_gradient(vel_j, conf, recorded_data; fre_range=fre_range, verbose=false);
        println("phi_j: ", phi_j, " phi_0 + c1*alpha_j*phi_0_diff: ", phi_0 + c1*alpha_j*phi_0_diff, " phi_lo: ", phi_lo)
        if (phi_j>(phi_0 + c1*alpha_j*phi_0_diff)) || (phi_j >= phi_lo)
            alpha_hi = alpha_j;
            println("alpha_hi = alpha_j")
        else
            phi_j_diff = sum((-1*phi_j./maximum(phi_j)) .* phi_j);
            println("phi_j_diff: ", phi_j_diff, " -c2 * phi_0_diff: ", -c2 * phi_0_diff);
            if (abs(phi_j_diff) <= -c2 * phi_0_diff)
                println("alpha is: ", alpha_j)
                alpha = alpha_j;
                break;
            end
            if phi_j_diff*(alpha_hi-alpha_lo) >= 0
                println("phi_j_diff*(alpha_hi-alpha_lo): ", phi_j_diff*(alpha_hi-alpha_lo), " hi and lo upsidedown.")
                alpha_hi = alpha_lo;
            end
            alpha_lo = alpha_j;
        end
    end
    if alpha == 0
        alpha = alpha_lo
        println("Zoom fail, alpha = alpha_lo: ", alpha);
    else
        println("Zoom succeed, alpha is: ", alpha)
    end
    return alpha
end

function line_search(vel_init, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, fre_range, vmin, vmax; c1 = 1e-11, c2=0.9, search_time=5, zoom_time=5)
    println("\nStart line search")
    phi_0_diff = sum(p_0 .* grad_0);

    iter = 1;
    alpha = 0;
    alpha_0 = 0;
    while iter <= search_time
        println("Search time: ", iter, " alpha_1: ", alpha_1)
        vel_1 = update_velocity(vel_init,alpha_1,p_0,vmin,vmax);
        grad_1, phi_1 = compute_gradient(vel_1, conf, recorded_data; fre_range=fre_range, verbose=false);
        println("phi_1: ", phi_1, " phi_0 + c1*alpha_1*phi_0_diff: ", phi_0 + c1*alpha_1*phi_0_diff, " phi_0: ",phi_0);
        if (phi_1>(phi_0 + c1*alpha_1*phi_0_diff)) || ((phi_1>=phi_0)&&(iter>1))
            alpha = zoomin(alpha_1, alpha_0, phi_0, vel_init, p_0, vmin, vmax, fre_range, zoom_time)
            break;
        end
        phi_1_diff = sum((-grad_1./maximum(grad_1)) .* grad_1)
        println("phi_1_diff: ", phi_1_diff, " -c2*phi_0_diff: ", -c2*phi_0_diff);
        if abs(phi_1_diff) <= -c2*phi_0_diff
            alpha = alpha_1;
            break;
        end
        if phi_1_diff >= 0
            alpha = zoomin(alpha_0, alpha_1, phi_1, vel_init, p_0, vmin, vmax, fre_range, zoom_time)
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
        alpha = alpha_1;
        println("Search fail, alpha is: ", alpha)
    else
        println("Search succeed, alpha is: ", alpha)
    end
    return alpha
end

alpha_1 = 300; alpha_max = 1000;
grad_0, phi_0 = compute_gradient(vel_init, conf, recorded_data; fre_range=[1], verbose=false);
p_0 = -grad_0./maximum(grad_0);
alpha = line_search(vel_init, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, [1], vmin, vmax; c1 = 1e-11, c2=0.9, search_time=5, zoom_time=5);
matshow(vel_new');colorbar()

grad, phi_0 = compute_gradient(vel_init, conf, recorded_data; fre_range=[1], verbose=true);
matshow(p'); colorbar()
p = -grad./maximum(grad);
phi_0_diff = sum(p .* grad)

alpha_1 = 600; alpha_max = 800;
alpha_0 = 0
c1 = 1e-11; c2 = 0.9

iter = 1;
vel_1 = update_velocity(vel_init,alpha_1,p,vmin,vmax);
grad_1, phi_1 = compute_gradient(vel_1, conf, recorded_data; fre_range=[1], verbose=true);

phi_1
phi_0 + c1*alpha_1*phi_0_diff
phi_0

if (phi_1>(phi_0 + c1*alpha_1*phi_0_diff)) || ((phi_1>=phi_0)&&(iter>1))
    # alpha = zoomin(alpha_1, alpha_0, phi_0, vel_init, p, vmin, vmax, [1])
    alpha = zoomin(alpha_1, 0, phi_0, vel_init, p, vmin, vmax, [1], 5)
    # break;
end

phi_1_diff = sum((-grad_1./maximum(grad_1)) .* grad_1)
c2*phi_0_diff
if abs(phi_1_diff) <= -c2*phi_0_diff
    alpha = alpha_1;
    # break;
end
if phi_1_diff >= 0
    alpha = zoomin(alpha_0, alpha_1, phi_1, vel_init, p, vmin, vmax, [1])
    # break;
end
alpha_0 = alpha_1
alpha_1 = 0.5 * (alpha_1 + alpha_max)

iter += 1;
