include("model_parameter.jl");
include("scalar_helmholtz_solver.jl");
using PyPlot;

omega = 2*pi*frequency;

ind_fre = 1;
@time beta, vel_ex = extend_area(vel_true, acq_fre);
@time A = make_diff_operator(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
@time B = make_diff_operator1(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
matshow(abs.(A-B));colorbar()


 # @code_warntype make_diff_operator(h,omega[ind_fre],vel_ex,beta,Nx_pml,Ny_pml);
matshow(real(A))
matshow(imag(A))

matshow(vel_ex')

vel = vel_ex;
omega = omega[1];
function make_diff_operator1(h,omega,vel_ex,beta,Nx_pml,Ny_pml)
    coef = (1 + im*beta) .* (h^2*omega.^2) ./ (vel_ex.^2);
    coef = coef - 4;
    coef0 = coef[2:end-1,2:end-1];
    coef0 = reshape(coef0, (Nx_pml-2)*(Ny_pml-2));
    vec1 = ones((Nx_pml-2)*(Ny_pml-2)-Nx_pml+2);
    vec2 = ones((Nx_pml-2)*(Ny_pml-2)-1);
    B = spdiagm((vec1, vec2, coef0, vec2, vec1), (-(Nx_pml-2),-1,0,1,(Nx_pml-2)));

    for i = 1:(Ny_pml-2-1)
        ind_x = (Nx_pml-2)*i+1;
        ind_y = (Nx_pml-2)*i;
        B[ind_x,ind_y] = 0;
        B[ind_y,ind_x] = 0;
    end
    return B
end
vel  = vel_ex;
coef = (1 + im*beta) .* (h^2*omega[1].^2) ./ (vel.^2);
coef = coef - 4;
coef0 = coef[2:end-1,2:end-1];
coef0 = reshape(coef0, (Nx_pml-2)*(Ny_pml-2));

vec1 = ones((Nx_pml-2)*(Ny_pml-2)-Nx_pml+2);
vec2 = ones((Nx_pml-2)*(Ny_pml-2)-1);
B = spdiagm((vec1, vec2, coef0, vec2, vec1), (-(Nx_pml-2),-1,0,1,(Nx_pml-2)));

for i = 1:(Ny_pml-2-1)
    ind_x = (Nx_pml-2)*i+1;
    ind_y = (Nx_pml-2)*i;
    B[ind_x,ind_y] = 0;
    B[ind_y,ind_x] = 0;
end
matshow(abs.(B-A)); colorbar()
