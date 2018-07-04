function l_BFGS(vel_init, conf, recorded_data, vmin, vmax; m=3, alpha_1=100, alpha_max=500, iter_time=5, c1=1e-11, c2=0.9, search_time=5, zoom_time=5, verbose=false, save_graph=false, fre_range="all")
	if iter_time <= (m+1)
		error("iter_time should be larger than m+1")
	end
	# Initialize
    iter_count = 0;
    vel_init = reshape(vel_init, conf.Nx*conf.Ny, 1);
    vel_0 = vel_init;

	# Main loop
	for ind_fre = 1:conf.fre_num
		iter_count += 1;
		fre_range1 = [ind_fre];
		S = zeros(conf.Nx*conf.Ny,m);
		Y = zeros(conf.Nx*conf.Ny,m);

		# First
	    println("\nIteration: ", iter_count, ", frequency: ", conf.frequency[ind_fre], ", before l-BFGS.")
	    grad_0, phi_0 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range1);
	    p_0 = -grad_0 / maximum(grad_0);
	    alpha = line_search(vel_0, conf, recorded_data, grad_0, p_0, phi_0, alpha_1, alpha_max, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)
	    vel_0 = update_velocity(vel_0,alpha,p_0,vmin,vmax);

		# Second
		iter_count += 1;
		println("\nIteration: ", iter_count, ", frequency: ", conf.frequency[ind_fre], ", before l-BFGS.")
		grad_1, phi_1 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range1);
		p_1 = -grad_1 / maximum(grad_1);
		alpha = line_search(vel_0, conf, recorded_data, grad_1, p_1, phi_1, alpha_1, alpha_max, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time)
		if alpha == 0
			println("Alpha is 0. Try to increase search time or alpha_1.")
			break;
		end
		vel_1 = update_velocity(vel_0,alpha,p_1,vmin,vmax);
		# Coef
		s = vel_1 - vel_0;
		y = grad_1 - grad_0;
		S[:,1] = s; Y[:,1] = y;
		# update
		vel_0[:] = vel_1; grad_1[:] = grad_0;

		# l-BFGS size
		m0 = 0;
		# l-BFGS iter
		for iter = 3:iter_time
			iter_count += 1;
			# update l-BFGS size
			if m0 < m
				m0 += 1;
			end
			println("\nIteration: ", iter_count, ", frequency: ", conf.frequency[ind_fre], ", l-BFGS", ", l-BFGS size is: ", m0)
			grad_1, phi_1 = compute_gradient(vel_0, conf, recorded_data; fre_range=fre_range1);
			q = grad_1;

			alpha_i_save = zeros(m0)
			for i = m0:-1:1
				rho_i = 1 ./ (Y[:,i].' * S[:,i]); rho_i = rho_i[1];
				alpha_i = rho_i * S[:,i].' * q; alpha_i = alpha_i[1];
				q = q - alpha_i * Y[:,i]
				alpha_i_save[i] = alpha_i;
			end
			gamma_k = (S[:,m0].' * Y[:,m0]) ./ (Y[:,m0].' * Y[:,m0]);
			gamma_k = gamma_k[1];
			H_0 = sparse(gamma_k*I, conf.Nx*conf.Ny, conf.Nx*conf.Ny);
			r = H_0 * q;
			for i = 1:m0
				rho_i = 1 ./ (Y[:,i].' * S[:,i]); rho_i = rho_i[1];
				beta = rho_i * Y[:,i].' * r; beta = beta[1];
				r = r + S[:,i]*(alpha_i_save[i] - beta)
			end
			# Descent direction
			p_1 = -r;

			# line search
			alpha = line_search(vel_0, conf, recorded_data, grad_1, p_1, phi_1, 1, 2, fre_range1, vmin, vmax; c1=c1, c2=c2, search_time=search_time, zoom_time=zoom_time);
			if alpha == 0
				println("Alpha is 0. Break.")
				@goto break_label
			end

			# update velocity
			vel_1 = update_velocity(vel_0,alpha,p_1,vmin,vmax);

			# update coef
			s = vel_1 - vel_0
			y = grad_1 - grad_0
			if m0 < m
				S[:,m0+1] = s; Y[:,m0+1] = y;
			else
				S[:,1:(m-1)] = S[:,2:m]
				Y[:,1:(m-1)] = Y[:,2:m]
				S[:,m] = s; Y[:,m] = y;
			end
			vel_0[:] = vel_1; grad_1[:] = grad_0;
		end

		@label break_label

		if save_graph == true
			title_name = conf.frequency[ind_fre];
			matshow((reshape(vel_0,conf.Nx,conf.Ny))',cmap="PuBu"); colorbar(); title("$title_name Hz")
			savefig("temp_graph/vel_$ind_fre.png");
			println("Velocity graph saved.")
		end
	end
	return vel_0;
end
