# FWI_fre
Frequency domain FWI code

- Based on Julia
- Frequency domain
- Use single core in most cases, efficiency for small size problem
- Easy for extend to parallel computing

Next stage task:
- Examine the Helmholtz differential operator size, change the shape of it
- Change the velocity and gradient to vector form
- Change the file arrangement and test 
- Write a new version line search method for steepest decent method and conjugate gradient method
- Add l-BFGS method
- Add truncated Newton method with l-BFGS preconditioning.
