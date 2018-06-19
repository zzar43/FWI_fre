# addprocs(2)

# include("model_parameter.jl");
using JLD2, PyPlot
@everywhere include("model_func.jl");
@everywhere include("scalar_helmholtz_solver.jl");
@everywhere include("FWI_fre.jl");

# ================================================
# Read data
@load "data/three_layers.jld2" vel_true vel_init Nx Ny h source_multi acq_fre
matshow((vel_true)', cmap="seismic"); colorbar();
matshow((vel_init)', cmap="seismic"); colorbar();
@load "data_compute/three_layers.jld2" wavefield_true recorded_data_true
# ================================================

c = 1e-6;
tau = 0.5;
search_time = 5;
alpha0 = 10;
vmin = 2;
vmax = 3;
iter_time = 0;
fre_record = [0];
misfit_diff_vec = [0];

iter_main = 1;
alpha = 0;
while (iter_main != length(acq_fre.frequency)) || (alpha != alpha0*tau^(search_time-1))
    println("=======================================")
    println("Main iteration time: ", iter_main, " frequency: ", acq_fre.frequency[iter_main], " alpha: ", alpha);

    iter_time += 1;
    fre_record = [fre_record, acq_fre.frequency[iter_main]];
    fre_range = iter_main:iter_main;

    # Compute gradient
    @time gradient, misfit_diff = compute_gradient_parallel(vel_init, recorded_data_true, source_multi, acq_fre, fre_range);

    # Compute direction
    p = -gradient./norm(gradient);
    # matshow((p)', cmap="seismic");colorbar();savefig("temp_graph/gradient_$iter_main.png")

    # line search
    # alpha = backtracking_line_search_parallel(vel_init,p,gradient,alpha0,misfit_diff,tau,c,search_time,fre_range,recorded_data_true,acq_fre,vmin,vmax);
    alpha = backtracking_line_search_parallel(vel_init,p,gradient,alpha0,misfit_diff,tau,c,search_time,"all",recorded_data_true,acq_fre,vmin,vmax);

    # update velocity
    if alpha == alpha0*tau^(search_time-1)
        iter_main += 1;
        # save graph
        matshow((vel_init)', cmap="seismic"); colorbar();savefig("temp_graph/vel_$iter_main.png")
    else
        vel_init = vel_init + alpha * p;
        vel_init[find(x->(x<vmin),vel_init)] = vmin;
        vel_init[find(x->(x>vmax),vel_init)] = vmax;
    end
    # matshow((vel_init)', cmap="seismic"); colorbar();savefig("temp_graph/vel_$iter_main.png")
end


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

matshow((vel_init)', cmap="gray"); colorbar();
