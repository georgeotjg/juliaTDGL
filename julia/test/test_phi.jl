# ---------------------------------------------------------------------------
#  Scalar potential coupled to the scheme.  Four controls; the third CAN
#  FAIL, and it catches errors of O(dt) that no log shows (an extra factor
#  `dt`, or a one-substep offset between the two halves of the gauge
#  transformation).
# ---------------------------------------------------------------------------
base_v14(; kw...) = Config(; GEOM_MODE="WEDGE", Nx_sc=16, Ny_sc=10, Nz_sc=4,
    dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30,
    gamma12=0.001, gamma_ratio=1.0, H_const=0.50, J_list=[0.08], delta_v=4,
    t_min_GL=0.0, t_min_GL_res=0.0, iter_min=120, mMax=120, use_Vconv=false,
    eps_conv=1e-14, eps_conv_seed=1e-14, selftest=false, out_dir="",
    verbose=false, kw...)

# The same four runs are needed by several @testset blocks.  Caching them
# does not change what is checked: each key is a different Config.
# (no `const`: this file is included inside a @testset, local scope)
_V14_cache = Dict{NamedTuple, SDE}()
function _v14_run(; use_phi=false, phi_bc=:dirichlet, phi_method=:separable)
    key = (; use_phi, phi_bc, phi_method)
    get!(_V14_cache, key) do
        run_case(base_v14(; use_phi, phi_bc, phi_method); mode=Strict())
    end
end

@testset "phi: the projection keeps div A = 0" begin
    for bc in (:dirichlet, :neumann)
        s = _v14_run(use_phi=true, phi_bc=bc)
        p = s.p
        dA = link_divergence(s.pos.estado, p)
        # strict interior only: that is where the projection imposes it
        int = view(dA, 3:p.Nx-1, 3:p.Ny-1, 3:p.Nz-1)
        # theta accumulates ~10 rad, so 1e-12 IS machine precision here
        @test maximum(abs, int) < 1e-10
        @test all(isfinite, s.pos.estado.G1)
    end
end

@testset "phi: Phi is a gauge freedom (the control that can fail)" begin
# |Psi| and h are GAUGE INVARIANT: they must come out EQUAL in the three
# gauges.  theta is not and must differ; if it were equal, the projection
# would be doing nothing.
    ref = _v14_run(use_phi=false).pos.estado
    for bc in (:dirichlet, :neumann)
        e = _v14_run(use_phi=true, phi_bc=bc).pos.estado
        @test maximum(abs, abs.(e.G1) .- abs.(ref.G1)) < 1e-13
        @test maximum(abs, abs.(e.G2) .- abs.(ref.G2)) < 1e-13
        @test maximum(abs, e.hz .- ref.hz)             < 1e-12
        @test maximum(abs, e.thetax .- ref.thetax)     > 1.0    # the gauge DOES change
    end
end

@testset "phi: the Josephson voltage does NOT depend on the gauge" begin
# The PHYSICAL voltage is invariant and is fully measured in the Phi = 0
# gauge.  What changes between gauges is how it is SPLIT between the drop
# of Phi and the drift of the phase; V_Phi alone, without the phase, is NOT
# an observable.
    r0 = _v14_run(use_phi=false).pos
    VJ = Float64[]; VP = Float64[]
    for bc in (:dirichlet, :neumann)
        r = _v14_run(use_phi=true, phi_bc=bc).pos
        push!(VJ, r.V_arr[end]); push!(VP, r.VPhi_arr[end])
    end
    # the Josephson voltage: identical in the three, to machine precision
    @test all(v -> abs(v - r0.V_arr[end]) <= 1e-12*max(abs(r0.V_arr[end]), 1e-12), VJ)
    # V_Phi: NOT invariant.  It depends on the BC by more than an order of magnitude.
    @test abs(VP[2]/VP[1]) > 5
    @test isnan(r0.VPhi_arr[end])            # no Phi, no V_Phi
    @info "phi: V_Josephson = $(r0.V_arr[end]) in the three gauges; V_Phi = $(VP) depending on the BC"
end

@testset "phi: a constant Phi changes nothing (trivial invariance)" begin
# A constant Phi cannot change any observable.  Checked directly on the
# block.
    cfg = base_v14(use_phi=true)
    p = derive_params(cfg); g = build_geometry(cfg, p); mode = Strict()
    sp = SolverPhi(p)
    s1 = initial_state(p, g, 0.5, mode)
    s2 = initial_state(p, g, 0.5, mode)
    for _ in 1:200
        substep_phi!(s1, p, g, mode, sp; H=0.5, HJ=0.0, J_cur=0.0)
    end
    # now with a Phi shifted by a constant at every substep
    sp2 = SolverPhi(p)
    for _ in 1:200
        psi_update!(s2.F1, s2.F2, s2.G1, s2.G2, s2.Vx, s2.Vy, s2.Vz, g.fmask, p, mode)
        TDGL.block_B!(s2, p, mode); TDGL.bloque_C!(s2, p, g)
        TDGL.block_C2!(s2, p, sp2)
        sp2.Phi .+= 0.017                       # arbitrary constant
        TDGL.apply_phi_psi!(s2, p, sp2, mode)
        TDGL.bloque_E!(s2, p, mode); TDGL.bloque_F!(s2, p)
        TDGL.bloque_G!(s2, p, 0.5, 0.0); TDGL.bloque_H!(s2, p, mode)
        TDGL.bloque_I!(s2, p)
    end
    @test maximum(abs, abs.(s1.G1) .- abs.(s2.G1)) < 1e-13
    @test maximum(abs, s1.hz .- s2.hz) < 1e-12
end

@testset "phi: F22 exposed as a choice, not inherited" begin
# At T = 0 the two branches agree TO THE BIT (which is how F22 went unseen).
    a = derive_params(base_v14(T_red=0.0, f22_supercurrent=true))
    b = derive_params(base_v14(T_red=0.0, f22_supercurrent=false))
    @test a.d1 === b.d1
    # At T != 0 they differ by exactly (1-T), which is mode F22.
    c = derive_params(base_v14(T_red=0.3, f22_supercurrent=true))
    d = derive_params(base_v14(T_red=0.3, f22_supercurrent=false))
    @test c.d1 ≈ 0.7*d.d1 rtol=1e-14
end

@testset "phi: the separable solver IS the LU, not an approximation" begin
    cfg = base_v14(use_phi=true)
    p = derive_params(cfg)
    for bc in (:dirichlet, :neumann)
        a = SolverPhi(p; bc=bc, metodo=:lu)
        b = SolverPhi(p; bc=bc, metodo=:separable)
        @test (a.n1, a.n2, a.n3) == (b.n1, b.n2, b.n3)
        r = randn(a.n1, a.n2, a.n3)
        bc === :neumann && (r .-= sum(r)/length(r))   # compatible with Neumann
        a.rhs .= r; solve!(a)
        b.rhs .= r; solve!(b)
        xa = copy(a.rhs); xb = copy(b.rhs)
        if bc === :neumann
            # with Neumann the two fix the constant differently (one node vs the
            # mean): compared after removing it, the only meaningful comparison
            # since that constant is gauge.
            xa .-= sum(xa)/length(xa); xb .-= sum(xb)/length(xb)
        end
        @test maximum(abs, xa .- xb) < 1e-11*max(maximum(abs, xa), 1.0)
    end
end

@testset "phi: the same run is bit for bit with itself" begin
# Forward anchor without a stored bank: if the Phi step changes by one ulp,
# this fails.  The bank data/v14_16x10x4/ (fixture_v14.jl) is the same run
# stored, for comparisons between machines.
    a = _v14_run(use_phi=true, phi_bc=:dirichlet)
    b = run_case(base_v14(use_phi=true, phi_bc=:dirichlet); mode=Strict())
    @test all(a.pos.estado.G1 .=== b.pos.estado.G1)
    @test all(a.pos.estado.thetax .=== b.pos.estado.thetax)
    @test a.pos.V_arr[end] === b.pos.V_arr[end]
    @test a.pos.VPhi_arr[end] === b.pos.VPhi_arr[end]
end

@testset "phi: the two methods give the SAME physics" begin
    ref = _v14_run(use_phi=true, phi_method=:lu).pos
    alt = _v14_run(use_phi=true, phi_method=:separable).pos
    @test maximum(abs, abs.(alt.estado.G1) .- abs.(ref.estado.G1)) < 1e-11
    @test maximum(abs, alt.estado.hz .- ref.estado.hz) < 1e-10
    @test abs(alt.V_arr[end] - ref.V_arr[end]) < 1e-10*max(abs(ref.V_arr[end]), 1e-10)
    @test abs(alt.VPhi_arr[end] - ref.VPhi_arr[end]) < 1e-8*max(abs(ref.VPhi_arr[end]), 1e-8)
end

@testset "phi: stored bank 16x10x4 Dirichlet (forward regression)" begin
# Dirichlet is the BC of the Octave scripts and of THIS bank.  It does not
# decide the physics: it decides which bits are compared against.
    ref_dir = joinpath(RAIZ, "data", "v14_16x10x4", "restart")
    @test isdir(ref_dir)
    sde = _v14_run(use_phi=true, phi_bc=:dirichlet, phi_method=:separable)
    campos = ("G1", "G2", "thetax", "thetay", "thetaz", "hx", "hy", "hz")
    for (pol, res) in (("Jpos", sde.pos), ("Jneg", sde.neg))
        ref = read_state(joinpath(ref_dir, "state_$pol.mat"))
        got = Dict("G1"=>res.estado.G1, "G2"=>res.estado.G2,
                   "thetax"=>res.estado.thetax, "thetay"=>res.estado.thetay,
                   "thetaz"=>res.estado.thetaz,
                   "hx"=>res.estado.hx, "hy"=>res.estado.hy, "hz"=>res.estado.hz)
        for c in campos
            @test all(ref[c] .=== got[c])
        end
        @test haskey(ref, "VPhi_arr")
        @test vec(Float64.(ref["VPhi_arr"]))[end] === res.VPhi_arr[end]
    end
    @test sde.pos.V_arr[end] ≈ 1.31217400e-03 rtol=1e-8
    @test sde.pos.VPhi_arr[end] ≈ 1.16334463e-02 rtol=1e-8
end
