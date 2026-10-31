# ============================================================================
#  materials.jl — material per region: band coefficients per node
#
#  A shape region with material = "name" carries the bands of a [materials.name] table of the run file
#  (the same keys as [material], in the units of band 1 of the first material).  Per node, the band update
#  reads its material's dt/Γ, ε = a − T, β, γ̃², Josephson row and diffusion factor rx/(m_ref Γh) from small
#  tables (materials × bands) through `mat[i,j,k]`; per link, a weight ρ = t·m_ref/m_link with
#  m_link = (m_a + m_b)/2 (the harmonic mean of the stiffnesses: the current crosses an interface as a
#  series resistance) and t the transparency of the pair of materials, times τ (0 across a free surface).
#  The supercurrent takes the same ρ, so the band equations and Ampère come from one functional.
#  A material equal to the first gives ρ = 1 and the same table entries to the bit, and the kernel's
#  expressions reduce exactly to today's (bands.jl): two regions of the same material are the uniform run.
#  κ and T stay those of the sample (the field and the bath).  σn may change by region: a link conducts with the
#  harmonic mean of its two nodes (series), and the field solve is then the PCG of the insulating padding (implicit.jl) even
#  without vacuum; the contacts and a metal padding take the σn of the face they touch.  A band can be absent from a material
#  (a one-band metal next to a two-band superconductor, `absent`): Ψ_µ = 0 there, the links into it carry no current of that
#  band (ρ_µ = 0), and it adds no stability bound.  Two normal fluids (qp_exchange) with materials: exchange.jl.
# ============================================================================

struct MatCoef{N}
    kx::Vector{Float64}; ky::Vector{Float64}; kz::Vector{Float64}     # rx/(m_ref Γh) of each material
    c::Vector{Float64}; ε::Vector{Float64}; β::Vector{Float64}; γ2::Vector{Float64}
    cpl::Vector{NTuple{N,Float64}}
end
_matcoef(b::AbstractVector) = MatCoef([x.kx for x in b], [x.ky for x in b], [x.kz for x in b], [x.c for x in b],
                                      [x.ε for x in b], [x.β for x in b], [x.γ2 for x in b], [x.cpl for x in b])

struct MatField{N}
    idx::Array{UInt8,3}                        # material of each node (1 = the first)
    models::Vector{Model{N}}                   # the materials, the first one is the Stepper's model
    full::NTuple{N,MatCoef{N}}                 # full-step coefficients per band
    stage::NTuple{N,MatCoef{N}}                # per (sub)stage, divided as BandCoef is
    ρ::NTuple{N,NTuple{3,Array{Float64,3}}}    # link weights per band and axis (τ included)
    t::Matrix{Float64}                         # transparency of each pair of materials
    absent::Matrix{Bool}                       # band µ absent from material m (materials × bands)
    absnodes::NTuple{N,Vector{CartesianIndex{3}}}   # per band, the interior nodes where it is absent (held at Ψ = 0)
    a::NTuple{N,Vector{Float64}}               # per band, the a of each material (the temperature field: ε = a − θ)
    DT::Vector{Float64}                        # D_T of the heat model per material (empty without heat)
end
_absent(mt::MatField, m, µ) = @inbounds mt.absent[m, µ]

# `absent`: nothing, a materials × bands Bool matrix, or a vector (per material) of the absent band indices
function _absent_matrix(absent, M, N)
    A = falses(M, N)
    absent === nothing && return Matrix{Bool}(A)
    if absent isa AbstractMatrix
        size(absent) == (M, N) || error("absent: a $M×$N matrix (materials × bands)")
        A .= absent
    else
        length(absent) == M || error("absent: one list of bands per material ($M)")
        for (m, l) in enumerate(absent), µ in l; A[m, µ] = true; end
    end
    any(A[1, :]) && error("absent: the first material ([material]) has all its bands")
    all(m -> !all(A[m, :]), 1:M) || error("absent: a material keeps at least one band")
    Matrix{Bool}(A)
end

function _check_materials(md::Model{N}, materials, mk, heat, drag, qp_exchange, noise, absent=nothing) where N
    mk.mat === nothing && error("materials: the shape names no material (material = \"name\" in a region)")
    mats = Model{N}[md]
    for m in materials
        m isa Model{N} || error("materials: every material needs $N bands")
        m.κ == md.κ && m.T == md.T ||
            error("materials: κ and T are those of the sample (the field and the bath); a material gives its bands, Josephson " *
                  "matrix and σn")
        m.σn > 0 || error("materials: σn must be positive")
        push!(mats, m)
    end
    A = _absent_matrix(absent, length(mats), N)
    for m in 2:length(mats), µ in 1:N
        A[m, µ] && eps_band(mats[m], µ) > 0 && error("absent: band $µ of material $m has ε = a − T > 0; an absent band has a <= T")
    end


    drag === nothing || error("materials: not with the drag yet")
    qp_exchange === nothing || qp_exchange isa Real ||
        error("materials: qp_exchange with materials is the two-band one (a number); N fluids (a matrix) are not supported per region")

    maximum(mk.mat) <= length(mats) || error("materials: the shape names material $(maximum(mk.mat)) of $(length(mats))")
    mats
end

function MatField(mats::Vector{Model{N}}, mk, gr::Grid, dt, rx, ry, rz, div::NTuple{N}; transparency=nothing,
                  absent=_absent_matrix(nothing, length(mats), N), heat_DT=nothing) where N
    M = length(mats)
    t = transparency === nothing ? ones(M, M) : Float64.(transparency)
    size(t) == (M, M) && t == transpose(t) && all(x -> 0 < x <= 1, t) && all(m -> t[m, m] == 1, 1:M) ||
        error("transparency: a symmetric $M×$M matrix in (0, 1] with ones on the diagonal")
    mref(µ) = mats[1].bands[µ].ma
    full = ntuple(µ -> _matcoef([_bandcoef(m, µ, dt, rx, ry, rz, mref(µ)) for m in mats]), N)
    stage = ntuple(N) do µ
        div[µ] == 1 ? full[µ] : _matcoef([_divided(_bandcoef(m, µ, dt, rx, ry, rz, mref(µ)), div[µ]) for m in mats])
    end
    idx = mk.mat
    function weights(µ, τ, o, ax)           # (the mass along the axis of the link: anisotropic bands)
        ρ = copy(τ); m0 = mats[1].bands[µ].ma[ax]
        for c in CartesianIndices(ρ)
            d = c + o
            checkbounds(Bool, idx, d) || continue
            a = idx[c]; b = idx[d]
            ma = mats[a].bands[µ].ma[ax]; mb = mats[b].bands[µ].ma[ax]
            ml = ma == mb ? ma : (ma + mb) / 2
            w = (ml == m0 ? 1.0 : m0 / ml) * (a == b ? 1.0 : t[a, b])
            ρ[c] = absent[a, µ] || absent[b, µ] ? 0.0 : τ[c] * w
        end
        ρ
    end
    ρ = ntuple(µ -> (weights(µ, mk.τx, CartesianIndex(1, 0, 0), 1), weights(µ, mk.τy, CartesianIndex(0, 1, 0), 2),
                     weights(µ, mk.τz, CartesianIndex(0, 0, 1), 3)), N)
    (; x1, x2, y1, y2, z1, z2) = gr
    absnodes = ntuple(µ -> [c for c in CartesianIndices((x1+1:x2, y1+1:y2, z1+1:z2)) if absent[idx[c], µ]], N)
    av = ntuple(µ -> [m.bands[µ].a for m in mats], N)
    MatField{N}(idx, mats, full, stage, ρ, t, absent, absnodes, av, heat_DT === nothing ? Float64[] : Float64.(heat_DT))
end

"""
    material_conductivity(gr, mk, mats, σvac, σref; cond=nothing) -> (Dx, Dy, Dz)

σ_l/σref on every link with σn per region: a conducting link carries the harmonic mean 2σ_aσ_b/(σ_a + σ_b) of its two nodes
(σ_a exactly when they agree), the node's σn being that of its material (a node outside the block takes the material of the
nearest block node: the contacts and a metal padding continue the face they touch).  With `σvac` the links of the insulating
padding (and of the vacuum of a sculpted sample, `cond`) are σvac/σref, as `link_conductivity`.
"""
function material_conductivity(gr::Grid, mk, mats, σvac, σref::Float64; cond=nothing)
    (; Nx, Ny, Nz, x1, x2, y1, y2, z1, z2) = gr
    base = σvac === nothing ? (ones(Nx, Ny+1, Nz+1), ones(Nx+1, Ny, Nz+1), ones(Nx+1, Ny+1, Nz)) :
           link_conductivity(gr, σvac, σref; cond=cond)
    σ(i, j, k) = @inbounds mats[mk.mat[clamp(i, x1, x2+1), clamp(j, y1, y2+1), clamp(k, z1, z2+1)]].σn
    function fill!(D, o)
        for c in CartesianIndices(D)
            D[c] == 1.0 || continue                     # insulating: σvac stays
            a = σ(c[1], c[2], c[3]); b = σ(c[1] + o[1], c[2] + o[2], c[3] + o[3])
            D[c] = (a == b ? a : 2a * b / (a + b)) / σref
        end
        D
    end
    (fill!(base[1], (1, 0, 0)), fill!(base[2], (0, 1, 0)), fill!(base[3], (0, 0, 1)))
end
