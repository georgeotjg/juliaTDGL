@testset "maxwell_implicit: implicit curl-curl in block [C]" begin
    kw = (; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
          kappa_s=25.0, kappa_p=25.0, xi_ratio=4.0, gamma12=1e-3, gamma_ratio=1.0,
          H_const=0.03, J_list=[0.0], selftest=false, snap_every=0, out_dir="",
          verbose=false, use_threads=false)
    # (1) off by default; on, the Maxwell bound leaves the step
    @test Config().maxwell_implicit == false
    p0 = derive_params(Config(; kw...)); p1 = derive_params(Config(; kw..., maxwell_implicit=true))
    @test occursin("Maxwell", p0.dt_limitante) && !occursin("Maxwell", p1.dt_limitante)
    @test p1.dt > 30 * p0.dt
    @test_throws ErrorException derive_params(Config(; kw..., maxwell_implicit=true, use_phi=true))
    # (2) the factorised operator is the explicit curl-curl term of [C], link by link,
    #     including the affine part from a tilt field
    p = derive_params(Config(; kw..., maxwell_implicit=true, h_tilt=0.01))
    g = build_geometry(Config(; kw...), p)
    mi = MaxwellImplicit(p, g); s = initial_state(p, g, 0.0, Strict())
    for f in (s.phix, s.phiy, s.phiz); f .= 0.1 .* reshape(sin.(1:length(f)), size(f)); end
    ox, oy, oz = similar(s.phix), similar(s.phiy), similar(s.phiz)
    TDGL._curl_term!(ox, oy, oz, s, p, 0.0, 0.0)
    pick(a, b, c) = vcat([a[I] for I in mi.ix], [b[I] for I in mi.iy], [c[I] for I in mi.iz])
    th = pick(s.phix, s.phiy, s.phiz); o = pick(ox, oy, oz)
    for f in (s.phix, s.phiy, s.phiz); fill!(f, 0.0); end
    TDGL._curl_term!(ox, oy, oz, s, p, 0.0, 0.0)
    @test maximum(abs, mi.L * th .+ pick(ox, oy, oz) .- o) < 1e-12 * maximum(abs, o)
    # (2b) the separable solver is chosen for dx = dy = dz and gives the LU step
    @test mi.sep !== nothing && mi.F === nothing
    mlu = MaxwellImplicit(p, g; solver=:lu)
    @test mlu.sep === nothing && mlu.F !== nothing
    s1 = initial_state(p, g, 0.0, Strict())
    for f in (s1.thetax, s1.thetay, s1.thetaz, s1.phix, s1.phiy, s1.phiz)
        f .= 0.1 .* reshape(cos.(1:length(f)) .* sin.(0.3 .* (1:length(f))), size(f))
    end
    s2 = deepcopy(s1)
    implicit_curl!(s1, mi); implicit_curl!(s2, mlu)
    @test maximum(abs, s1.phix .- s2.phix) < 1e-12 && maximum(abs, s1.phiy .- s2.phiy) < 1e-12 &&
          maximum(abs, s1.phiz .- s2.phiz) < 1e-12
    # (3) the same fixed point: the J = 0 seed at kappa = 25 relaxed with both schemes
    #     (measured: |Psi| to 3e-8, hz to 2e-11, M to 8 digits, at a 38 times larger step)
    estados = map((false, true)) do imp
        c = Config(; kw..., maxwell_implicit=imp, t_min_GL=0.0, iter_min=1, mMax=10^5,
                   eps_conv_seed=1.0, tie_seed_tolerance=true, eps_conv_rate=1e-7)
        q = derive_params(c); gg = build_geometry(c, q)
        TDGL.sweep([0.0], [0.03], "J+", gg, q, Strict())
    end
    a, b = estados
    @test maximum(abs, abs.(a.estado.G1) .- abs.(b.estado.G1)) < 1e-6
    @test maximum(abs, abs.(a.estado.G2) .- abs.(b.estado.G2)) < 1e-6
    @test maximum(abs, a.estado.hz .- b.estado.hz) < 1e-8
    @test abs(a.M_arr[1] - b.M_arr[1]) < 1e-7
end
