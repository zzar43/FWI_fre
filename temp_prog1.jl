# addprocs(2);
using JLD2, PyPlot;
@everywhere include("inverse_problem.jl");

# Load data
@load "data/overthrust_small.jld2" vel_true vel_init conf
@load "data_compute/overthrust_small.jld2" recorded_data
vmin = minimum(vel_true);
vmax = maximum(vel_true);
matshow((vel_true)',cmap="PuBu",clim=[vmin,vmax]); colorbar(); title("True model");
savefig("temp_graph/vel_true.png");
matshow((vel_init)',cmap="PuBu",clim=[vmin,vmax]); colorbar(); title("Initial model");
savefig("temp_graph/vel_init.png");

vmax = maximum(vel_true);
vmin = minimum(vel_true);
alpha_max = 500;
alpha_1 = 20;

# grad, phi = compute_gradient(vel_init, conf, recorded_data; fre_range=[4]);
# p = -grad ./ maximum(grad);
# matshow(p'); colorbar();

fre_range = [7]
for i = 1:10
    println("Iter: ", i)
    grad, phi = compute_gradient(vel_init, conf, recorded_data; fre_range=fre_range);
    p = -grad ./ maximum(grad);
    alpha = line_search(vel_init, conf, recorded_data, grad, p, phi, alpha_1, alpha_max, fre_range, vmin, vmax; c1 = 1e-11, c2=0.9, search_time=5, zoom_time=6)
    vel_init = update_velocity(vel_init,alpha,p,vmin,vmax);
end

matshow((vel_init)',cmap="PuBu",clim=[vmin,vmax]); colorbar(); title("Initial model");
vel_back = vel_init;
@save "temp_data" vel_init

aalpha = linspace(10,1*19,20);
phi = zeros(20);
phi_diff = zeros(20);
phi_diff1 = zeros(20);

grad, phii = compute_gradient(vel_init, conf, recorded_data; fre_range=[3]);
p = -grad ./ maximum(grad);

for i = 1:20
    vel_new = update_velocity(vel_init,aalpha[i],p,vmin,vmax);
    grad, misfit = compute_gradient(vel_new, conf, recorded_data; fre_range=[1]);
    phi[i] = misfit
    phi_diff[i] = sum(grad .* grad) / maximum(grad);
    phi_diff1[i] = sum(grad .* grad) / norm(grad);
    println(i)
end

matshow(p');colorbar()
plot(aalpha,phi_diff)
plot(aalpha,phi)
