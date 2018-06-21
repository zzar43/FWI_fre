struct acquisition_fre
    # space
    Nx::Int64
    Ny::Int64
    h::Float32
    # time
    Nt::Int64
    dt
    t
    # frequency
    frequency::Array{Float32}
    fre_num::Int64
    # source
    source_num::Int64
    source_coor
    source_multi
    # receiver
    receiver_num::Int64
    receiver_coor
    projection_op
    projection_op_pml
    # PML
    pml_len::Int64
    pml_alpha::Float32
    Nx_pml::Int64
    Ny_pml::Int64
end
