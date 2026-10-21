@testset "pi_substeps: multirate substeps of the pi band" begin
    kw = (; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
          kappa_s=2.0, kappa_p=2.0, xi_ratio=4.0, gamma12=1e-3, gamma_ratio=1.0,
          H_const=0.03, J_list=[0.0], selftest=false, snap_every=0, out_dir="",
          verbose=false, use_threads=false, cfl_legacy=false, maxwell_implicit=true)
    # (1) off by default; automatic: the pi bound no longer binds, each substep stays below it
    @test Config(:legacy).pi_substeps == 1
    p1 = derive_params(Config(:legacy; kw...)); pa = derive_params(Config(:legacy; kw..., pi_substeps=0))
    @test p1.n_pi == 1 && isempty(p1.Wpi) && occursin("pi", p1.dt_limitante)
    @test pa.n_pi > 1 && occursin("sigma", pa.dt_limitante)
    @test pa.dt / pa.n_pi < p1.dt / 0.99
    @test_throws ErrorException derive_params(Config(:legacy; kw..., pi_substeps=-1))
    @test_throws ErrorException derive_params(Config(:legacy; kw..., pi_substeps=2, use_drag=true, nu_drag=0.1))
    # automatic substeps with a term: one substep, the implicit curl-curl stays; Phi is not implicit
    for t in ((; use_heat=true), (; T_red=0.3, T_physical=true), (; use_drag=true, nu_drag=0.05))
        q = derive_params(Config(:prod; kw..., t...))
        @test q.n_pi == 1 && q.maxwell_implicit
    end
    cφ = Config(:prod; kw..., use_phi=true); derive_params(cφ)          # Φ: the explicit scheme, by itself
    @test !cφ.maxwell_implicit && cφ.pi_substeps == 1
    # (2) one step with n substeps = n steps of the plain kernel at dt/n for pi, with sigma
    #     and the links frozen and the pi ghosts of block [H] refreshed in between
    n = 4
    pn = derive_params(Config(:legacy; kw..., pi_substeps=n))
    pd = derive_params(Config(:legacy; kw..., dt_scale=min(1.0, pn.dt / n / p1.dt)))
    @test pd.dt ≈ pn.dt / n
    g = build_geometry(Config(:legacy; kw...), pn); s = initial_state(pn, g, 0.0, Strict())
    for (k, f) in enumerate((s.Vx, s.Vy, s.Vz))
        f .= cis.(0.3 .* reshape(sin.(k .+ (1:length(f))), size(f)))
    end
    s.G1 .*= cis.(0.2 .* reshape(cos.(1:length(s.G1)), size(s.G1)))
    s.G2 .*= (1 .+ 0.1 .* reshape(sin.(0.7 .* (1:length(s.G2))), size(s.G2))) .*
             cis.(0.5 .* reshape(sin.(0.3 .* (1:length(s.G2))), size(s.G2)))
    TDGL.ghosts_pi!(s.G2, s.Vx, s.Vy, s.Vz, pn, Strict())
    (; x1, x2, y1, y2, z1, z2) = pn
    I = (x1+1:x2, y1+1:y2, z1+1:z2)
    # previous-step link rotation dtheta (theta - phi); zero = frozen links
    for (dth, extrap) in ((0.0, true), (0.01, true), (0.01, false), (0.3, true))   # 0.3: exact branch
        q = derive_params(Config(:legacy; kw..., pi_substeps=n, pi_link_extrapolation=extrap))
        ss = deepcopy(s)
        D = [dth .* reshape(cos.((1:length(V)) .+ c), size(V)) for (c, V) in enumerate((ss.Vx, ss.Vy, ss.Vz))]
        for (U, V, d) in zip((ss.Ux, ss.Uy, ss.Uz), (ss.Vx, ss.Vy, ss.Vz), D)
            U .= V .* cis.(d)                            # previous links: V = U exp(-i dtheta)
        end
        G2_0 = copy(ss.G2)
        psi_update!(ss.F1, ss.F2, ss.G1, ss.G2, ss.Vx, ss.Vy, ss.Vz, g.fmask, q, Strict())
        TDGL.pi_substeps!(ss, q, g.fmask, Strict())
        @test ss.G2 == G2_0                              # block [B] reads G2 after the update
        L = [copy(ss.Vx), copy(ss.Vy), copy(ss.Vz)]
        R = [cis.(-(extrap ? d : 0 .* d) ./ n) for d in D]
        X = copy(ss.G2); Y1 = similar(ss.G1); Y2 = similar(ss.G2)
        for m in 1:n
            psi_update!(Y1, Y2, ss.G1, X, L..., g.fmask, pd, Strict())
            for c in 1:3; L[c] .*= R[c]; end
            X .= Y2; TDGL.ghosts_pi!(X, L..., pn, Strict())
        end
        @test maximum(abs, ss.F2[I...] .- X[I...]) < (extrap && dth > 0 ? 1e-8 : 1e-13)   # series below |dth| = 0.05, O(dth^5)
    end
    # (2b) sigma substeps: the same for sigma, pi frozen in the coupling
    ps = derive_params(Config(:legacy; kw..., sigma_substeps=n, pi_substeps=0, dt_scale=0.5))
    @test ps.n_sig == n && ps.n_pi > 1                       # pi follows the larger step
    qd = derive_params(Config(:legacy; kw..., pi_substeps=0, dt_scale=0.5))   # sigma at dt/n, plain
    @test qd.dt ≈ ps.dt / n
    ss = deepcopy(s)
    for (U, V) in zip((ss.Ux, ss.Uy, ss.Uz), (ss.Vx, ss.Vy, ss.Vz))
        U .= V .* cis.(0.02 .* reshape(sin.(1:length(V)), size(V)))
    end
    TDGL.ghosts_pi!(ss.G1, ss.Vx, ss.Vy, ss.Vz, ps, Strict())
    psi_update!(ss.F1, ss.F2, ss.G1, ss.G2, ss.Vx, ss.Vy, ss.Vz, g.fmask, ps, Strict())
    TDGL.pi_substeps!(ss, ps, g.fmask, Strict())
    L = [copy(ss.Vx), copy(ss.Vy), copy(ss.Vz)]
    R = [cis.(-angle.(conj.(V .* conj.(U))) ./ n) for (V, U) in zip((ss.Vx, ss.Vy, ss.Vz), (ss.Ux, ss.Uy, ss.Uz))]
    X = copy(ss.G1); Y1 = similar(ss.G1); Y2 = similar(ss.G2)
    for m in 1:n
        psi_update!(Y1, Y2, X, ss.G2, L..., g.fmask, qd, Strict())
        for c in 1:3; L[c] .*= R[c]; end
        X .= Y1; TDGL.ghosts_pi!(X, L..., ps, Strict())
    end
    @test maximum(abs, ss.F1[I...] .- X[I...]) < 1e-10
    # (2c) with substeps psi_update! computes only the band without them, with the bits of _A_plano!
    for q in (pn, ps)
        A1 = similar(s.G1); A2 = similar(s.G2); B1 = similar(s.G1); B2 = similar(s.G2)
        fill!(A1, 0); fill!(A2, 0); fill!(B1, 0); fill!(B2, 0)
        psi_update!(A1, A2, s.G1, s.G2, s.Vx, s.Vy, s.Vz, g.fmask, q, Strict())
        for k in q.z1+1:q.z2; TDGL._A_plano!(B1, B2, s.G1, s.G2, s.Vx, s.Vy, s.Vz, g.fmask, q, Strict(), k); end
        @test q.n_sig > 1 || A1[I...] == B1[I...]
        @test q.n_pi > 1 || A2[I...] == B2[I...]
    end
    # (3) the same fixed point: the J = 0 seed relaxed with and without substeps
    estados = map((1, 0)) do ns
        c = Config(:legacy; kw..., pi_substeps=ns, t_min_GL=0.0, iter_min=1, mMax=10^5,
                   eps_conv_seed=1.0, tie_seed_tolerance=true, eps_conv_rate=1e-7)
        q = derive_params(c); gg = build_geometry(c, q)
        TDGL.sweep([0.0], [0.03], "J+", gg, q, Strict())
    end
    a, b = estados
    @test maximum(abs, abs.(a.estado.G1) .- abs.(b.estado.G1)) < 1e-6
    @test maximum(abs, abs.(a.estado.G2) .- abs.(b.estado.G2)) < 1e-6
    @test abs(a.M_arr[1] - b.M_arr[1]) < 1e-7
    # (4) Config(:prod) and the restart: with the previous-step link phases in the
    #     checkpoint a resumed prod run is the uninterrupted one, to the bit
    cp = Config(:prod; kw...)
    @test cp.maxwell_implicit && !cp.cfl_legacy && cp.pi_substeps == 0 && cp.dt_scale == 0.5 &&
          cp.conv_by_time && cp.blas_threads == 1
    @test_throws ErrorException Config(:nope)
    dir = mktempdir(); mkpath(joinpath(dir, "restart"))
    qp = derive_params(Config(:prod; kw..., out_dir=dir)); gp = build_geometry(Config(:prod; kw...), qp)
    mip = MaxwellImplicit(qp, gp); sa = initial_state(qp, gp, 0.03, Strict())
    paso!(x) = substep!(x, qp, gp, Strict(); H=0.03, HJ=0.0, J_cur=0.4, mi=mip)
    for _ in 1:60; paso!(sa); end
    save_checkpoint(qp, sa, "J+", 1, [0.0], [0.4])
    sb, = load_checkpoint(checkpoint_path(joinpath(dir, "restart"), "Jpos"), qp, Strict())
    for _ in 1:60; paso!(sa); paso!(sb); end
    @test sa.G1 == sb.G1 && sa.G2 == sb.G2 && sa.thetax == sb.thetax && sa.hz == sb.hz
end
