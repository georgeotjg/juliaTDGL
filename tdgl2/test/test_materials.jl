# Material per region (materials.jl): the uniform path through the material tables, the interface on the lattice, the
# run file.
using TDGL, Test, Random, TOML

_withmat(mk, f, gr) = (mat = ones(UInt8, size(mk.f)); for c in CartesianIndices(mat); f(c) && (mat[c] = 0x02); end;
                       TDGL._ghost_planes!(mat, gr); Masks(mk.f, mk.τx, mk.τy, mk.τz, mk.τψ, mk.vac, mat))
_copy(md) = Model(md.bands, [md.γ[i][j] for i in 1:nbands(md), j in 1:nbands(md)]; κ=md.κ, σn=md.σn, T=md.T)
_same(a, b) = all(µ -> a.G[µ] == b.G[µ], 1:length(a.G)) && a.φx == b.φx && a.φy == b.φy && a.φz == b.φz && a.hz == b.hz

@testset "materials: two regions of the same material are the uniform run, to the bit" begin
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    for (md, kw) in ((two_band(xi_ratio=4.0, κ=2.0), (;)), (two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0), (;)),
                     (two_band(xi_ratio=4.0, κ=25.0), (rkl=false,)))
        mk2 = _withmat(mk, c -> c[1] > gr.x1 + 12, gr)
        a = Stepper(md, gr, mk; threads=false, kw...); b = Stepper(md, gr, mk2; threads=false, materials=[_copy(md)], kw...)
        sa = initial_state(md, gr, mk, 0.03); sb = initial_state(md, gr, mk2, 0.03; materials=b.mt.models)
        HJ = selffield(a, 0.5)
        for _ in 1:150; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
        @test a.rkl == b.rkl && a.dt == b.dt && _same(sa, sb)
    end
end

@testset "materials: the interface on the lattice" begin
    # the discrete continuity, node by node: (Γ_n/dt) Im(Ḡ ΔΨ) = div J/dx² through an S/S′ interface (m, a, Γ all differ)
    gr = Grid(12, 6, 4; dx=0.5); mk = box_masks(gr); mk2 = _withmat(mk, c -> c[1] >= gr.x1 + 6, gr)
    md = Model((Band(),), zeros(1, 1); κ=2.0); m′ = Model((Band(m=1.5, a=0.6, Γ=1.3),), zeros(1, 1); κ=2.0)
    st = Stepper(md, gr, mk2; threads=false, implicit=false, substeps=(1,), materials=[m′])
    s = initial_state(md, gr, mk2, 0.0; materials=st.mt.models); G = s.G[1]; rng = Xoshiro(3)
    for c in eachindex(G); mk.τψ[c] > 0 && (G[c] *= (0.7 + 0.3rand(rng)) * cis(2randn(rng))); end
    TDGL.neumann!(s, st); TDGL._band_pass!(s.F[1], G, s, 1, s.Vx, s.Vy, s.Vz, st.bc[1], st, nothing); TDGL.supercurrent!(s, st)
    (; x1, x2, y1, y2, z1, z2, dx) = gr; r = 0.0
    for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        Γn = st.mt.models[mk2.mat[i,j,k]].bands[1].Γ
        lhs = (Γn / st.dt) * imag(conj(G[i,j,k]) * (s.F[1][i,j,k] - G[i,j,k]))
        rhs = ((s.Jx[i,j,k] - s.Jx[i-1,j,k]) + (s.Jy[i,j,k] - s.Jy[i,j-1,k]) + (s.Jz[i,j,k] - s.Jz[i,j,k-1])) / dx^2
        r = max(r, abs(lhs - rhs))
    end
    @test r < 1e-12
    # the proximity decay into a normal S′ (a′ = −0.5, m′ = 2) against the linearised solution on the lattice
    gr = Grid(80, 2, 2; dx=0.5); mk = box_masks(gr); i0 = gr.x1 + 40; mk2 = _withmat(mk, c -> c[1] >= i0, gr)
    m′ = Model((Band(m=2.0, a=-0.5),), zeros(1, 1); κ=2.0)
    st = Stepper(md, gr, mk2; threads=false, materials=[m′])
    s = initial_state(md, gr, mk2, 0.0; materials=st.mt.models)
    for _ in 1:round(Int, 300 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
    ψ = abs.(s.G[1][:, 2, 2]); ξ = 1.0; ℓd = 0.5 / acosh(1 + 0.5 * 0.25 / (2 * 0.5))
    i3 = i0 + round(Int, 3ξ / 0.5); i6 = i0 + round(Int, 6ξ / 0.5)
    @test abs((i6 - i3) * 0.5 / log(ψ[i3] / ψ[i6]) / ℓd - 1) < 1e-3
    # a strip all of S′ through the material path is the uniform run of S′ (the step of S′ in both: the first is softer)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    md = Model((Band(m=1.2, a=0.8), Band(m=1 / (16 * 0.5), a=0.5)), [0.0 1e-3; 1e-3 0.0]; κ=2.0)
    m′ = Model((Band(), Band(m=1 / (16 * 0.59), a=0.59)), [0.0 1e-3; 1e-3 0.0]; κ=2.0)
    mk2 = _withmat(mk, c -> true, gr)
    a = Stepper(md, gr, mk2; threads=false, materials=[m′]); b = Stepper(m′, gr, mk; threads=false)
    sa = initial_state(md, gr, mk2, 0.03; materials=a.mt.models); sb = initial_state(m′, gr, mk, 0.03); HJ = selffield(b, 0.4)
    for _ in 1:500; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
    @test a.dt == b.dt && maximum(µ -> maximum(abs.(sa.G[µ] .- sb.G[µ])), 1:2) < 1e-12
    # the mirror y → −y with an x slab of S′: +J is the mirror of −J
    md = two_band(xi_ratio=4.0, κ=2.0); gr = Grid(16, 8, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="SIM"), gr); mk2 = _withmat(mk, c -> gr.x1 + 6 <= c[1] <= gr.x1 + 10, gr)
    m′ = Model((Band(m=1.5, a=0.7), md.bands[2]), [0.0 1e-3; 1e-3 0.0]; κ=2.0)
    run(J) = (st = Stepper(md, gr, mk2; threads=false, materials=[m′]); s = initial_state(md, gr, mk2, 0.05; materials=st.mt.models);
              HJ = selffield(st, J); for _ in 1:round(Int, 5 / st.dt); step!(s, st; H=0.05, HJ=HJ); end; s)
    p = run(0.3); q = run(-0.3); ny = size(p.G[1], 2)
    @test all(maximum(abs.(abs.(p.G[µ]) .- abs.(q.G[µ][:, ny:-1:1, :]))) < 1e-12 for µ in 1:2)
end

@testset "materials: the energy balance with an interface" begin
    function balance(sc)
        md = two_band(xi_ratio=4.0, κ=2.0); gr = Grid(16, 10, 8; dx=0.5); mk = box_masks(gr); ghost_links_off!(mk, gr)
        mk2 = _withmat(mk, c -> c[1] >= gr.x1 + 8, gr)
        m′ = Model((Band(m=2.0, a=0.6, Γ=1.5), Band(m=1 / (16 * 0.5), a=0.5)), [0.0 3e-3; 3e-3 0.0]; κ=2.0)
        st = Stepper(md, gr, mk2; threads=false, implicit=false, substeps=(1, 1), dt_scale=sc, materials=[m′])
        s = initial_state(md, gr, mk2, 0.0); rng = Xoshiro(7)
        for µ in 1:2, c in eachindex(s.G[µ]); mk.τψ[c] > 0 && (s.G[µ][c] *= (0.6 + 0.4rand(rng)) * cis(1.5randn(rng))); end
        for _ in 1:round(Int, 0.5 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
        r = 0.0; mono = true
        for _ in 1:round(Int, 0.2 / st.dt)
            s0 = deepcopy(s); F0 = free_energy(s0, st); step!(s, st; H=0.0, HJ=0.0); F1 = free_energy(s, st)
            D, W = dissipation(s0, s, st); r = max(r, abs(((F1 - F0) / st.dt + D - W) / D)); mono &= F1 < F0
        end
        r, mono
    end
    a = balance(0.5); b = balance(0.25)
    @test a[2] && b[2] && a[1] < 3e-3 && 0.4 < b[1] / a[1] < 0.6        # gradient flow, residual O(dt)
end

@testset "materials: the run file and the guards" begin
    c = load_config(joinpath(@__DIR__, "..", "examples", "bicapa_tunel.toml"))
    @test c.geom.materials == ["lenta"] && count(==(0x02), c.masks.mat) > 0 && c.scheme[:transparency][1, 2] == 0.2
    st = Stepper(c.model, c.grid, c.masks; c.scheme..., threads=false)
    k = (c.grid.z1 + c.grid.z2) ÷ 2
    ks = [kk for kk in c.grid.z1+1:c.grid.z2 if c.masks.mat[20, 8, kk] != c.masks.mat[20, 8, kk+1]]
    @test length(ks) == 1 && st.mt.ρ[1][3][20, 8, ks[1]] ≈ 0.2 * 1.0 / ((1.0 + 2.0) / 2)       # t · m_ref/m_link
    @test st.mt.ρ[1][3][20, 8, ks[1] + 1] == 0.5 && st.mt.ρ[1][3][20, 8, ks[1] - 1] == 1.0    # inside each slab: m_ref/m
    @test_throws ErrorException Stepper(c.model, c.grid, c.masks; c.scheme..., threads=false, qp_exchange=1.0)
    @test_throws ErrorException Stepper(c.model, c.grid, c.masks; threads=false)              # the shape names a material
    bad = Model(c.model.bands, [0.0 1e-3; 1e-3 0.0]; κ=3.0)
    @test_throws ErrorException Stepper(c.model, c.grid, c.masks; c.scheme..., threads=false, materials=[bad])
    g = Grid(8, 4, 4; dx=0.5); f = tempname() * ".toml"
    write(f, "[[shape]]\ntype = \"rect\"\nx = 1.0\ny = 1.0\nw = 1.0\nh = 1.0\nmaterial = \"nada\"\n")
    @test_throws ErrorException build_masks(GeomSpec(GEOM_MODE="CUSTOM", geom_file=f, materials=["otro"]), g)
end

@testset "materials: RKC on one band, three materials, the explicit keys" begin
    # rkl = "rkc" with σ on RKL stages and π on RKC (an Int and a Float64 divisor of the stage tables): the uniform run to the bit
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    md = two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0); mk2 = _withmat(mk, c -> c[1] > gr.x1 + 12, gr)
    a = Stepper(md, gr, mk; threads=false, rkl="rkc"); b = Stepper(md, gr, mk2; threads=false, rkl="rkc", materials=[_copy(md)])
    @test a.rkc[1] === nothing && a.rkc[2] !== nothing
    sa = initial_state(md, gr, mk, 0.03); sb = initial_state(md, gr, mk2, 0.03; materials=b.mt.models); HJ = selffield(a, 0.5)
    for _ in 1:100; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
    @test a.rkl == b.rkl && _same(sa, sb)
    # three regions, materials 2 and 3 copies of the first: the uniform run to the bit
    m0 = two_band(xi_ratio=4.0, κ=2.0)
    mat = ones(UInt8, size(mk.f)); mat[gr.x1+9:end, :, :] .= 0x02; mat[gr.x1+17:end, :, :] .= 0x03; TDGL._ghost_planes!(mat, gr)
    mk3 = Masks(mk.f, mk.τx, mk.τy, mk.τz, mk.τψ, nothing, mat)
    a = Stepper(m0, gr, mk; threads=false); b = Stepper(m0, gr, mk3; threads=false, materials=[_copy(m0), _copy(m0)])
    sa = initial_state(m0, gr, mk, 0.03); sb = initial_state(m0, gr, mk3, 0.03; materials=b.mt.models)
    for _ in 1:100; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
    @test _same(sa, sb)
    # three distinct materials (m, a, Γ) and a 3×3 transparency: the discrete continuity node by node, and the weight of each interface
    g1 = Grid(18, 6, 4; dx=0.5); mb = box_masks(g1)
    mat = ones(UInt8, size(mb.f)); mat[g1.x1+7:end, :, :] .= 0x02; mat[g1.x1+13:end, :, :] .= 0x03; TDGL._ghost_planes!(mat, g1)
    mk3 = Masks(mb.f, mb.τx, mb.τy, mb.τz, mb.τψ, nothing, mat)
    q1 = Model((Band(),), zeros(1, 1); κ=2.0); q2 = Model((Band(m=1.5, a=0.6, Γ=1.3),), zeros(1, 1); κ=2.0)
    q3 = Model((Band(m=0.7, a=0.8, Γ=0.8),), zeros(1, 1); κ=2.0); t = [1 0.5 0.3; 0.5 1 0.2; 0.3 0.2 1]
    st = Stepper(q1, g1, mk3; threads=false, implicit=false, substeps=(1,), materials=[q2, q3], transparency=t)
    @test st.mt.ρ[1][1][g1.x1+6, 3, 3] ≈ 0.5 / ((1 + 1.5) / 2) && st.mt.ρ[1][1][g1.x1+12, 3, 3] ≈ 0.2 / ((1.5 + 0.7) / 2)
    s = initial_state(q1, g1, mk3, 0.0; materials=st.mt.models); G = s.G[1]; rng = Xoshiro(5)
    for c in eachindex(G); mb.τψ[c] > 0 && (G[c] *= (0.7 + 0.3rand(rng)) * cis(2randn(rng))); end
    TDGL.neumann!(s, st); TDGL._band_pass!(s.F[1], G, s, 1, s.Vx, s.Vy, s.Vz, st.bc[1], st, nothing); TDGL.supercurrent!(s, st)
    (; x1, x2, y1, y2, z1, z2, dx) = g1; r = 0.0
    for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        Γn = st.mt.models[mat[i,j,k]].bands[1].Γ
        lhs = (Γn / st.dt) * imag(conj(G[i,j,k]) * (s.F[1][i,j,k] - G[i,j,k]))
        rhs = ((s.Jx[i,j,k] - s.Jx[i-1,j,k]) + (s.Jy[i,j,k] - s.Jy[i,j-1,k]) + (s.Jz[i,j,k] - s.Jz[i,j,k-1])) / dx^2
        r = max(r, abs(lhs - rhs))
    end
    @test r < 1e-12
    # the run file: a key that [material] gives (gamma_kwt, the coupling) left out of a [materials.<name>] table is an error, not
    # the package default; ghost_links = false turns the face links off
    src = read(joinpath(@__DIR__, "..", "examples", "bicapa_tunel.toml"), String)
    ld(txt) = (f = tempname() * ".toml"; write(f, txt); load_config(f))
    @test_throws ErrorException ld(replace(src, "gamma12 = 1e-3\ntransparency" => "transparency"))
    kw = replace(src, "kappa = 2.0\n" => "kappa = 2.0\nband = [{ name = \"sigma\", m = 1.0, a = 1.0, gamma_kwt = 10.0 }, " *
                 "{ name = \"pi\", xi_ratio = 4.0, a = 0.59, gamma_kwt = 10.0 }]\n"; count=1)
    @test_throws ErrorException ld(kw)
    @test ld(replace(kw, "a = 0.8\n" => "a = 0.8\ngamma_kwt = 10.0\n", "a = 0.5\n" => "a = 0.5\ngamma_kwt = 10.0\n")).scheme[:materials][1].bands[1].γkwt == 10.0
    # a table may take a material file of materials/: its bands and coupling, with the sample's κ, T and (unless given) σn
    i = findfirst("\n[materials.lenta]", src).start + 1; j = findfirst("\n[mesh]", src).start + 1
    cf = ld(src[1:i-1] * "[materials.lenta]\nfile = \"mgb2_dirty.toml\"\ntransparency = { material = 0.2 }\n\n" * src[j:end])
    mf = cf.scheme[:materials][1]
    @test mf.κ == cf.model.κ && mf.σn == cf.model.σn && mf.bands[2].a < 0 && mf.γ[1][2] ≈ 1.886
    c0 = ld(src); c1 = ld(replace(src, "buf = 1\n" => "buf = 1\nghost_links = false\n"))
    @test sum(c0.masks.τx) > sum(c1.masks.τx) && all(c1.masks.τx[:, c1.grid.y1, c1.grid.z1] .== 0)
end

# The run file of a bilayer stopped by STOP.txt after its first point and resumed is the run without a stop, to the bit (rungs,
# state), as the menu runs it; its folder keeps the scheme defaults, and summary.json the materials and the integrators.
@testset "materials: stop and resume a run file to the bit" begin
    src = read(joinpath(@__DIR__, "..", "examples", "bicapa_tunel.toml"), String)
    txt = replace(src, "nx = 40" => "nx = 20", "ny = 16" => "ny = 8", "nz = 12" => "nz = 8", "J = [0.30, 0.50, 0.70]" => "J = [0.0, 0.3, 0.5]",
                  "t_min_GL = 200.0" => "t_min_GL = 10.0", "t_min_GL_res = 400.0" => "t_min_GL_res = 20.0")
    d = mktempdir(); f = joinpath(d, "b.toml"); write(f, txt)
    a = joinpath(d, "a"); b = joinpath(d, "b"); mkpath(b)
    ra = run_config(f; out_dir=a, polarity=:pos)
    write(joinpath(b, "STOP.txt"), "stop\n")
    run_config(f; out_dir=b, polarity=:pos)
    n1 = countlines(joinpath(b, "output", "rungs_J+.csv"))
    mv(joinpath(b, "STOP.txt"), joinpath(mktempdir(), "STOP.txt"))
    rb = run_config(f; out_dir=b, resume=true, polarity=:pos)
    @test n1 < countlines(joinpath(a, "output", "rungs_J+.csv"))                       # it did stop early
    @test read(joinpath(a, "output", "rungs_J+.csv"), String) == read(joinpath(b, "output", "rungs_J+.csv"), String)
    @test ra.pos.state.G[1] == rb.pos.state.G[1] && ra.pos.state.G[2] == rb.pos.state.G[2] && ra.pos.state.φx == rb.pos.state.φx
    @test TOML.parsefile(joinpath(b, "scheme_defaults.toml"))["rkl"] == "rkc"
end

# σn per region, two normal fluids with materials, a one-band metal.
@testset "materials: σn per region, qp_exchange, a one-band metal" begin
    cm(m; σn=m.σn) = Model(m.bands, [m.γ[i][j] for i in 1:nbands(m), j in 1:nbands(m)]; κ=m.κ, σn=σn, T=m.T)
    md = two_band(xi_ratio=4.0, κ=2.0)
    # the normal state of an S(σ₁)/S(σ₂) strip (Ψ held at 0): the steady E_x in series, E₂/E₁ = σ₁/σ₂
    gr = Grid(40, 4, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="SIM"), gr); mk.f .= 0.0
    mk2 = _withmat(mk, c -> c[1] >= gr.x1 + 20, gr)
    st = Stepper(md, gr, mk2; threads=false, hard=true, materials=[cm(md; σn=0.25)])
    @test st.imp isa TDGL.PCGSolver
    s = initial_state(md, gr, mk2, 0.0; materials=st.mt.models); HJ = selffield(st, 0.1)
    for _ in 1:round(Int, 150 / st.dt); step!(s, st; H=0.0, HJ=HJ); end
    θ0 = copy(s.θx); step!(s, st; H=0.0, HJ=HJ)
    E(i) = -(s.θx[i, 3, 3] - θ0[i, 3, 3]) / st.dt
    @test isapprox(E(gr.x2 - 6) / E(gr.x1 + 6), 4; rtol=1e-6)
    # equal σn: the separable solve and the bits; qp_exchange with a copy of the material: the bits
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    mk2 = _withmat(mk, c -> c[1] > gr.x1 + 12, gr)
    for kw in ((;), (qp_exchange=1.0,))
        a = Stepper(md, gr, mk; threads=false, kw...); b = Stepper(md, gr, mk2; threads=false, materials=[cm(md)], kw...)
        @test b.imp isa TDGL.SepSolver
        sa = initial_state(md, gr, mk, 0.03); sb = initial_state(md, gr, mk2, 0.03; materials=b.mt.models); HJ = selffield(a, 0.5)
        for _ in 1:100; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
        @test _same(sa, sb)
    end
    # a different exchange rate per material goes through the variable-coefficient solve; a split that changes is refused
    b = Stepper(md, gr, mk2; threads=false, qp_exchange=1.0, materials=[cm(md)], qp_rates=[1.0, 10.0])
    @test b.qpx.var !== nothing
    s = initial_state(md, gr, mk2, 0.03); HJ = selffield(b, 0.5)
    for _ in 1:50; step!(s, b; H=0.03, HJ=HJ); end
    @test all(isfinite, s.G[1]) && b.qpx.ps.iters[1] > 0
    other = Model((md.bands[1], Band(m=2 * md.bands[2].m, a=0.59)), [0.0 1e-3; 1e-3 0.0]; κ=2.0)
    @test_throws ErrorException Stepper(md, gr, mk2; threads=false, qp_exchange=1.0, materials=[other])
    # a one-band metal (band 2 absent): Ψπ = 0 there, the same stages, δ = 0 there with qp_exchange
    metal = Model((Band(a=-0.5), Band(m=md.bands[2].m, a=-1.0)), [0.0 0.0; 0.0 0.0]; κ=2.0)
    for kw in ((;), (qp_exchange=1.0,))
        st = Stepper(md, gr, mk2; threads=false, materials=[metal], absent=[Int[], [2]], kw...)
        @test st.rkl == Stepper(md, gr, mk; threads=false, kw...).rkl
        s = initial_state(md, gr, mk2, 0.03; materials=st.mt.models); HJ = selffield(st, 0.2)
        for _ in 1:200; step!(s, st; H=0.03, HJ=HJ); end
        inN = [c for c in CartesianIndices(s.G[2]) if mk2.mat[c] == 0x02 && mk.τψ[c] > 0]
        @test all(c -> s.G[2][c] == 0, inN) && maximum(c -> abs(s.G[1][c]), inN) > 0.1
        haskey(kw, :qp_exchange) && @test all(c -> st.qpx.δ[c] == 0, [c for c in inN if c[1] <= gr.Nx])
    end
    @test_throws ErrorException Stepper(md, gr, mk2; threads=false, materials=[cm(md)], absent=[Int[], [2]])   # a > 0 absent
end

# noise, anisotropic bands and heat with materials (drag stays refused).
@testset "materials: noise, anisotropic bands, heat" begin
    cm(m; bands=m.bands) = Model(bands, [m.γ[i][j] for i in 1:nbands(m), j in 1:nbands(m)]; κ=m.κ, σn=m.σn, T=m.T)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    mk2 = _withmat(mk, c -> c[1] > gr.x1 + 12, gr)
    function pair(md; n=100, kw...)
        a = Stepper(md, gr, mk; threads=false, kw...); b = Stepper(md, gr, mk2; threads=false, materials=[cm(md)], kw...)
        sa = initial_state(md, gr, mk, 0.03); sb = initial_state(md, gr, mk2, 0.03; materials=b.mt.models); HJ = selffield(a, 0.5)
        for _ in 1:n; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
        _same(sa, sb) && (a.heat === nothing || a.heat.θ == b.heat.θ)
    end
    md = two_band(xi_ratio=4.0, κ=2.0)
    @test pair(md; noise=1e-3, noise_seed=5)
    # noise with KWT (RKC stages: SK-ROCK and the log-polar step): the material path is the uniform one
    @test pair(two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0); n=50, noise=1e-3, noise_seed=5)
    @test pair(Model((Band(m_axes=(1.0, 1.0, 2.0)), Band(m=0.106, a=0.59)), [0.0 1e-3; 1e-3 0.0]; κ=2.0))
    @test pair(md; heat=Heat(D_T=0.25, joule=true))
    # the continuity node by node through an interface of two anisotropic materials
    g = Grid(12, 6, 4; dx=0.5); m = box_masks(g); m2 = _withmat(m, c -> c[1] >= g.x1 + 6, g)
    a1 = Model((Band(m_axes=(1.0, 2.0, 0.5)),), zeros(1, 1); κ=2.0); a2 = Model((Band(m_axes=(1.5, 0.7, 3.0), a=0.6, Γ=1.3),), zeros(1, 1); κ=2.0)
    st = Stepper(a1, g, m2; threads=false, implicit=false, substeps=(1,), materials=[a2])
    s = initial_state(a1, g, m2, 0.0; materials=st.mt.models); G = s.G[1]; rng = Xoshiro(3)
    for c in eachindex(G); m.τψ[c] > 0 && (G[c] *= (0.7 + 0.3rand(rng)) * cis(2randn(rng))); end
    TDGL.neumann!(s, st); TDGL._band_pass!(s.F[1], G, s, 1, s.Vx, s.Vy, s.Vz, st.bc[1], st, nothing); TDGL.supercurrent!(s, st)
    (; x1, x2, y1, y2, z1, z2, dx) = g; r = 0.0
    for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        lhs = (st.mt.models[m2.mat[i,j,k]].bands[1].Γ / st.dt) * imag(conj(G[i,j,k]) * (s.F[1][i,j,k] - G[i,j,k]))
        rhs = ((s.Jx[i,j,k] - s.Jx[i-1,j,k]) + (s.Jy[i,j,k] - s.Jy[i,j-1,k]) + (s.Jz[i,j,k] - s.Jz[i,j,k-1])) / dx^2
        r = max(r, abs(lhs - rhs))
    end
    @test r < 1e-12
    # D_T per material: the heat content conserved to rounding (no bath, no source)
    g = Grid(40, 4, 2; dx=0.5); m = box_masks(g); m.f .= 0.0; m2 = _withmat(m, c -> c[1] > g.x1 + 20, g)
    st = Stepper(md, g, m2; threads=false, hard=true, heat=Heat(D_T=0.25, gamma_bath=0.0, eta=0.0), materials=[cm(md)], heat_DT=[0.25, 1.0])
    s = initial_state(md, g, m2, 0.0; materials=st.mt.models)
    T = st.heat.θ; for c in CartesianIndices(T); T[c] = c[1] <= g.x1 + 10 ? 0.4 : 0.1; end
    I = CartesianIndices((g.x1+1:g.x2, g.y1+1:g.y2, g.z1+1:g.z2)); S0 = sum(T[I])
    for _ in 1:2000; step!(s, st; H=0.0, HJ=0.0); end
    @test abs(sum(T[I]) / S0 - 1) < 1e-12
    @test_throws ErrorException Stepper(md, gr, mk2; threads=false, materials=[cm(md)], drag=0.05)          # drag: refused
end
