# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);
# matshow((vel_true)',cmap="PuBu",clim=[vmin,vmax]); colorbar(); title("True model");
# savefig("temp_graph/vel_true.png");
# matshow((vel_init)',cmap="PuBu",clim=[vmin,vmax]); colorbar(); title("Initial model");
# savefig("temp_graph/vel_init.png");
alpha_1 = 30;
alpha_max = 500;
fre_range = [4];
m = 3;
S = zeros(conf.Nx*conf.Ny,m);
Y = zeros(conf.Nx*conf.Ny,m);

iter = 1;
vel_init = reshape(vel_init, conf.Nx*conf.Ny, 1);
grad_0, phi_0 = compute_gradient(vel_init, conf, recorded_data; fre_range=fre_range);
p_0 = -grad_0 / maximum(grad_0);
alpha = line_search(vel_init, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, fre_range, vmin, vmax; c1=1e-11, c2=0.9, search_time=8, zoom_time=4)
vel_0 = update_velocity(vel_init,alpha,p_0,vmin,vmax);

for iter = 2:(m+1)
    println(iter)
    grad_1, phi_1 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range);
    p_1 = -grad_1 / maximum(grad_1);
    alpha = line_search(vel_0, conf, recorded_data, grad_1, p_1, phi_1, alpha_1, alpha_max, fre_range, vmin, vmax; c1=1e-11, c2=0.9, search_time=8, zoom_time=4);
    if alpha == 0
        println("Alpha is 0. Try to increase search time or alpha_1.")
        break;
    end
    vel_1 = update_velocity(vel_0,alpha,p_1,vmin,vmax);
    # Coef
    s = vel_1 - vel_0;
    y = grad_1 - grad_0;
    S[:,iter-1] = s; Y[:,iter-1] = y;
    # update
    vel_0[:] = vel_1; grad_1[:] = grad_0;
end
vel_back = vel_0;

for iter = m+2:1:10
    println(iter)
    grad_1, phi_1 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range);
    q = grad_1;
    alpha_i_save = zeros(m)
    for i = m:-1:1
        rho_i = 1 ./ (Y[:,i].' * S[:,i]); rho_i = rho_i[1];
        alpha_i = rho_i * S[:,i].' * q; alpha_i = alpha_i[1];
        q = q - alpha_i * Y[:,i]
        alpha_i_save[i] = alpha_i;
    end
    gamma_k = (S[:,m].' * Y[:,m]) ./ (Y[:,m].' * Y[:,m]);
    gamma_k = gamma_k[1];
    H_0 = sparse(gamma_k*I, conf.Nx*conf.Ny, conf.Nx*conf.Ny);
    r = H_0 * q;
    for i = 1:m
        rho_i = 1 ./ (Y[:,i].' * S[:,i]); rho_i = rho_i[1];
        beta = rho_i * Y[:,i].' * r; beta = beta[1];
        r = r + S[:,i]*(alpha_i_save[i] - beta)
    end
    # Descent direction
    p_1 = -r;
    # line search
    alpha = line_search(vel_0, conf, recorded_data, grad_1, p_1, phi_1, 1, alpha_max, fre_range, vmin, vmax; c1=1e-11, c2=0.9, search_time=8, zoom_time=8);
    if alpha == 0
        println("Alpha is 0. Break.")
        break;
    end
    # update velocity
    vel_1 = update_velocity(vel_0,alpha,p_1,vmin,vmax);
    # update coef
    s = vel_1 - vel_0
    y = grad_1 - grad_0
    S[:,1:(m-1)] = S[:,2:m]
    Y[:,1:(m-1)] = Y[:,2:m]
    S[:,3] = s; Y[:,3] = y;
    vel_0[:] = vel_1; grad_1[:] = grad_0;
end

vel = reshape(vel_0, conf.Nx, conf.Ny);
matshow(vel', cmap="PuBu"); colorbar()
