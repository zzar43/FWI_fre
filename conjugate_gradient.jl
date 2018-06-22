# addprocs(2)

# include("model_parameter.jl");
using JLD2, PyPlot
# @everywhere include("model_func.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");
@everywhere include("model_func.jl");
@everywhere include("optimization.jl")

# ================================================
# Read data
@load "data/three_layers.jld2" vel_true vel_init Nx Ny h source_multi acq_fre
@load "data_compute/three_layers.jld2" wavefield_true recorded_data_true
matshow((vel_true)', cmap="GnBu"); colorbar();
matshow((vel_init)', cmap="GnBu"); colorbar();
# ================================================
# For three layers
c = 5e-6;
tau = 0.5;
search_time = 4;
alpha0 = 8.0;
vmin = minimum(vel_true);
vmax = maximum(vel_true);
iter_time = 0;
fre_record = [0];
misfit_diff_vec = [0];
thres = 0.02

# For Marmousi
# c = 5e-3;
# tau = 0.5;
# search_time = 6;
# alpha0 = 20000;
# vmin = minimum(vel_true);
# vmax = maximum(vel_true);
# iter_time = 0;
# fre_record = [0];
# misfit_diff_vec = [0];
th = zeros(length(acq_fre.frequency));
for ind_fre = 1:acq_fre.fre_num
    for ind_source = 1:acq_fre.source_num
        th[ind_fre] += 0.5*norm(recorded_data_true[:,ind_fre,ind_source]);
    end
end
iter_time = 1;
iter_main = 1;
alpha = 0;
s0 = zeros(Float32, Nx, Ny);
s1 = zeros(Float32, Nx, Ny);
while (iter_main <= length(acq_fre.frequency))
    println("=======================================")
    println("Main iteration time: ", iter_time, " frequency: ", acq_fre.frequency[iter_main], " alpha: ", alpha);
    iter_time += 1;
    fre_range = iter_main:iter_main;

    # Compute gradient
    @time gradient, misfit_diff = compute_gradient_parallel(vel_init, recorded_data_true, source_multi, acq_fre, fre_range);
    # Compute direction
    p0 = -gradient./norm(gradient);
    s0[:] = p0;
    # line search
    alpha, misfit_diff_new = backtracking_line_search_parallel(vel_init,p0,gradient,alpha0,tau,c,search_time,fre_range,recorded_data_true,acq_fre,vmin,vmax);
    # update velocity
    println("misfit_diff_new/th[iter_main]: ", misfit_diff_new/th[iter_main])
    if misfit_diff_new/th[iter_main] < thres
        iter_main += 1;
        # save graph
        matshow((vel_init)', cmap="GnBu"); colorbar();savefig("temp_graph/vel_$iter_main.png")
    else
        vel_init = vel_init + alpha * p0;
        vel_init[find(x->(x<vmin),vel_init)] = vmin;
        vel_init[find(x->(x>vmax),vel_init)] = vmax;
    end
    count_conj = 1;
    misfit_diff_new0 = 0;
    alpha_cg = alpha0;
    while misfit_diff_new/th[iter_main] > thres
        println("-----------------------------------------")
        count_conj += 1;
        gradient, misfit_diff = compute_gradient_parallel(vel_init, recorded_data_true, source_multi, acq_fre, fre_range);
        p1 = -gradient./norm(gradient);
        beta = sum(p1.*(p1-p0)) / sum(p0.*p0);
        beta = max(beta,0);
        s1[:] = p1 + beta * s0;
        alpha, misfit_diff_new = backtracking_line_search_parallel(vel_init,s1./(norm(s1)),gradient,alpha_cg,tau,c,search_time,fre_range,recorded_data_true,acq_fre,vmin,vmax);
        if (alpha == 0) || (misfit_diff_new > misfit_diff_new0)
            break;
        end
        alpha_cg = alpha;
        # update velocity
        println("misfit_diff_new/th[iter_main]: ", misfit_diff_new/th[iter_main])
        if (misfit_diff_new/th[iter_main] < thres)
            iter_main += 1;
            # save graph
            matshow((vel_init)', cmap="GnBu"); colorbar();savefig("temp_graph/vel_$iter_main.png");
            break;
        else
            vel_init = vel_init + alpha * s1;
            vel_init[find(x->(x<vmin),vel_init)] = vmin;
            vel_init[find(x->(x>vmax),vel_init)] = vmax;
        end
        # update coef
        s0[:] = s1; p0[:] = p1;
        println("CG process: ", count_conj, " beta: ", beta);
    end
    # record misfit function
    # misfit_diff = compute_misfit_func(vel_init, source_multi, acq_fre, recorded_data_true);
    # misfit_diff_vec = [misfit_diff_vec misfit_diff];
    # matshow((vel_init)', cmap="seismic"); colorbar();savefig("temp_graph/vel_$iter_main.png")
end

misfit_diff_vec = misfit_diff_vec[2:end];

# iter_main = 1;
# for iter_main = 1:iter_time
#     println("=======================================")
#     println("Main iteration time: ", iter_main)
#     # Compute gradient
#     @time gradient, misfit_diff = compute_gradient_parallel(vel_init, recorded_data_true, source_multi, acq_fre, fre_range[iter_main]);
#     misfit_diff_vec[iter_main] = misfit_diff;
#     # Compute direction
#     p = -gradient./norm(gradient);
#     matshow((p)', cmap="seismic");colorbar();savefig("temp_graph/gradient_$iter_main.png")
#     # line search
#     # alpha = backtracking_line_search_parallel(vel_init,p,gradient,alpha0,misfit_diff,tau,c,search_time,fre_range[iter_main],recorded_data_true,acq_fre,vmin,vmax);
#     alpha = backtracking_line_search_parallel(vel_init,p,gradient,alpha0,misfit_diff,tau,c,search_time,"all",recorded_data_true,acq_fre,vmin,vmax);
#     # update velocity
#     vel_init = vel_init + alpha * p;
#     vel_init[find(x->(x<vmin),vel_init)] = vmin;
#     vel_init[find(x->(x>vmax),vel_init)] = vmax;
#
#     matshow((vel_init)', cmap="seismic"); colorbar();savefig("temp_graph/vel_$iter_main.png")
# end

matshow((vel_init + 100000*p)', cmap="seismic"); colorbar();
plot(misfit_diff_vec)
