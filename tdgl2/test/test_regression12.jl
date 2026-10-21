# The N-band step against juliaTDGL 1.2 (Fast mode), to the bit, with N = 2 and every new
# parameter neutral.  Golden states: test/golden/make_golden.jl.
using JLD2, LinearAlgebra

const GOLDEN = joinpath(@__DIR__, "golden")

function _load_case(nombre)
    d = JLD2.load(joinpath(GOLDEN, "$nombre.jld2"))
    d["par"], d["ini"], d["fin"], Masks(d["fmask"], d["taux"], d["tauy"], d["tauz"], d["taupsi"])
end

function _stepper_12(par, mk; threads=false)
    md = two_band(xi_ratio=sqrt(1 / (par["m_ratio"] * par["a2n"])), a2n=par["a2n"],
                  gamma12=par["gamma12"], gamma_ratio=par["gamma_ratio"], κ=par["kappa"])
    # m_r exactly as 1.2 had it (the xi_ratio round trip may move its last bit)
    b1, b2 = md.bands
    md = Model((b1, Band(m=par["m_ratio"], Γ=b2.Γ, a=b2.a, β=b2.β)),
               [0.0 par["gamma12"]; par["gamma12"] 0.0]; κ=par["kappa"])
    gr = Grid(par["Nx_sc"], par["Ny_sc"], par["Nz_sc"]; dx=par["dx"], buf=par["buf"])
    impl = par["implicit"]
    terms = get(par, "heat", false) || get(par, "drag", 0.0) != 0
    sub = (par["pi_substeps_cfg"] == 1 || terms) ? (1, 1) : (1, 0)   # 1.2 takes one pi substep with terms on
    heat = get(par, "heat", false) ? Heat(D_T=par["D_T"], gamma_bath=par["gamma_bath"], eta=par["eta_J"],
                                          joule=par["joule"], margin=par["heat_margin"]) : nothing
    drag = get(par, "drag", 0.0) != 0 ? par["drag"] : nothing
    Stepper(md, gr, mk; implicit=impl, substeps=sub, dt_scale=par["dt_scale"],
            extrap=par["extrap"], threads=threads, heat=heat, drag=drag, selffield_bc=:slab, rkl=false)
end

function _state_from(gr, d)
    s = State(gr, (d["G1"], d["G2"]), (d["thetax"], d["thetay"], d["thetaz"]),
              (d["phix"], d["phiy"], d["phiz"]), (d["hx"], d["hy"], d["hz"]))
    s
end

# The golden states were written on OpenBLAS's Haswell kernels with Julia code for that CPU.  By default every
# entry must agree to 1e-10 of max(1, |entry|): on other kernels and CPUs the dense products of the separable
# solve and the SIMD reductions round differently (measured: ~1e-15 on the states, up to 1.0e-12 on ladder
# voltages, which are time averages over many steps, on an AVX-512 machine).  Verdicts and counts stay exact.
# TDGL_TEST_BITS=1 asks for the bit, which holds with OPENBLAS_CORETYPE=Haswell and `julia -C haswell` on any
# x86 machine with AVX2 (the pinned job of the CI does that).
const _BITS = get(ENV, "TDGL_TEST_BITS", "") == "1"
const _BLAS_CORE = try
    lib = Base.Libc.Libdl.dlopen(BLAS.get_config().loaded_libs[1].libname)
    unsafe_string(ccall(Base.Libc.Libdl.dlsym(lib, :openblas_get_corename64_), Cstring, ()))
catch
    "?"
end
@info _BITS ? "regression against 1.2 to the bit (OpenBLAS kernel $(repr(_BLAS_CORE)))" :
              "regression against 1.2 to 1e-10 (TDGL_TEST_BITS=1 for the bit; OpenBLAS kernel $(repr(_BLAS_CORE)))"
function _same(a, b)
    size(a) == size(b) || return false
    _BITS && return all(reinterpret(UInt64, vec(collect(a))) .== reinterpret(UInt64, vec(collect(b))))
    d = maximum(abs.(collect(a) .- collect(b)) ./ max.(1.0, abs.(collect(b))); init=0.0)
    d <= 1e-10 || @info "off by $d (scaled by max(1, |b|))"
    d <= 1e-10
end

@testset "regression against 1.2: $nombre" for nombre in ("k2_R13_explicit", "k2_R4_prod", "k25_R4_prod",
                                                          "k2_R4_heat", "k2_R4_drag")
    par, ini, fin, mk = _load_case(nombre)
    st = _stepper_12(par, mk)
    @test st.dt === par["dt"]
    @test st.nsub == (1, par["n_pi"])
    @test selffield(st, par["J"]) === par["HJ"]
    s = _state_from(st.grid, ini)
    st.heat === nothing || (st.heat.θ .= ini["Temp"])
    # the links of the state are exp(-i θ) of its phases, as 1.2 had them
    @test _same(s.Vx, ini["Vx"]) && _same(s.Ux, ini["Ux"]) && _same(s.Uz, ini["Uz"])
    for _ in 1:par["N1"]
        step!(s, st; H=par["H"], HJ=par["HJ"])
    end
    @test _same(s.G[1], fin["G1"])
    @test _same(s.G[2], fin["G2"])
    @test _same(s.θx, fin["thetax"]) && _same(s.θy, fin["thetay"]) && _same(s.θz, fin["thetaz"])
    @test _same(s.φx, fin["phix"]) && _same(s.φy, fin["phiy"]) && _same(s.φz, fin["phiz"])
    @test _same(s.Vx, fin["Vx"]) && _same(s.Vy, fin["Vy"]) && _same(s.Vz, fin["Vz"])
    @test _same(s.hx, fin["hx"]) && _same(s.hy, fin["hy"]) && _same(s.hz, fin["hz"])
    st.heat === nothing || @test _same(st.heat.θ, fin["Temp"]) && maximum(st.heat.θ) > 1e-4
end

# Same bits with any number of threads (run the suite with `julia -t 4` to exercise it).
if Threads.nthreads() > 1
    @testset "threads ($(Threads.nthreads())): $nombre" for nombre in ("k2_R4_prod", "k25_R4_prod")
        par, ini, fin, mk = _load_case(nombre)
        st = _stepper_12(par, mk; threads=true)
        s = _state_from(st.grid, ini)
        for _ in 1:par["N1"]
            step!(s, st; H=par["H"], HJ=par["HJ"])
        end
        @test _same(s.G[1], fin["G1"]) && _same(s.G[2], fin["G2"]) && _same(s.θx, fin["thetax"])
    end
end
