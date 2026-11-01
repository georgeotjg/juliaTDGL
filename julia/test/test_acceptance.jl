# ---------------------------------------------------------------------------
#  ACCEPTANCE: the stored reference states are reproduced bit for bit.
#  Target: max|Delta| = 0, not "close".
#
#  If this ever stops giving exactly zero, do NOT relax the tolerance: find
#  the cause.  Each time a 1e-15 appeared there was a concrete cause (hypot,
#  sincos and the FMA contraction of the complex product), not "rounding noise".
# ---------------------------------------------------------------------------
@testset "acceptance: bit for bit against the reference states" begin
    salida = mktempdir()
    cfg = cfg_fixture(out_dir=salida, run_tag="test_aceptacion", selftest=true)
    sde = run_case(cfg; mode=Strict())

    # the point closes where the reference says it closes
    @test sde.pos.m_arr[1] == 40
    @test sde.pos.cflag[1] == 'F'
    @test sde.pos.r1_arr[1] ≈ 9.424e-06 rtol=1e-3
    @test sde.pos.M_arr[1] ≈ -1.419e-01 rtol=1e-3
    @test sde.pos.Nv1_arr[1] == 0 && sde.pos.Nv2_arr[1] == 0
    @test sde.pos.ps1_arr[1] ≈ 0.8702 atol=1e-4
    @test sde.pos.ps2_arr[1] ≈ 0.4880 atol=1e-4

    campos = ("G1", "G2", "thetax", "thetay", "thetaz", "hx", "hy", "hz")
    peor = 0.0
    for pol in ("Jpos", "Jneg")
        ref = read_state(joinpath(FIXT, "restart", "state_$pol.mat"))
        got = read_state(checkpoint_path(joinpath(salida, "restart"), pol))
        for c in campos
            @test size(ref[c]) == size(got[c])
            @test all(ref[c] .=== got[c])          # === compares BITS, not values
            peor = max(peor, maximum(abs, ref[c] .- got[c]))
        end
    end
    @test peor == 0.0
    @info "acceptance: worst max|Δ| over 16 arrays = $peor"
end

@testset "acceptance: bank 2, transport with vortex flow" begin
    # Bank 1 is pure Meissner: V=0, Nv=0, and with delta_v=8 on Nx_sc=12 the
    # voltmeter planes CROSS (vi=10 > vo=5), so the phase path is EMPTY.  A
    # code could pass it with no voltmeter at all.  This bank covers that.
    salida = mktempdir()
    cfg = Config(:legacy; GEOM_MODE="WEDGE", Nx_sc=20, Ny_sc=14, Nz_sc=4,
                 dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0,
                 xi_ratio=1.30, gamma12=0.001, gamma_ratio=1.3, H_const=0.50,
                 J_list=[0.10], delta_v=6, t_min_GL=0.0, t_min_GL_res=0.0,
                 iter_min=400, mMax=400, use_Vconv=false,
                 eps_conv=1e-14, eps_conv_seed=1e-14, selftest=false,
                 snap_every=0, out_dir=salida, run_tag="transporte",
                 verbose=false)
    sde = run_case(cfg; mode=Strict())
    ref_dir = joinpath(RAIZ, "data", "transport_20x14x4", "restart")

    campos = ("G1", "G2", "thetax", "thetay", "thetaz", "hx", "hy", "hz")
    for pol in ("Jpos", "Jneg")
        ref = read_state(joinpath(ref_dir, "state_$pol.mat"))
        got = read_state(checkpoint_path(joinpath(salida, "restart"), pol))
        for c in campos
            @test all(ref[c] .=== got[c])
        end
    end

    # The OBSERVABLES are not in the checkpoint: they exercise the voltmeter
    # (gauge-invariant phase unwrapped line by line, slip count, slope fit)
    # and the vortex count.
    r = sde.pos
    @test r.m_arr[2] == 401 && r.cflag[2] == '*'
    @test r.V_arr[2]     ≈  1.5102e-01 rtol=1e-4      # V_dc
    @test r.nslip_arr[2] ≈  86.13      rtol=1e-3      # phase slips
    @test r.Vst_arr[2]   ≈  5.0972e-01 rtol=1e-4      # static drop
    @test r.M_arr[2]     ≈ -1.4684e-01 rtol=1e-4      # magnetisation
    @test r.V2_arr[2]    ≈  1.4631e-01 rtol=1e-4      # voltage of the pi band
    @test r.Nv1_arr[2] == 1 && r.Nv2_arr[2] == 1
    # the diode: the reverse polarity does not dissipate
    @test abs(sde.neg.V_arr[2]) < 1e-4
    # second-half slope (diagnostic for the verdict rule): finite on every closed point,
    # same sign as V on the flowing one, small on the reverse one
    @test all(isfinite, r.Vh_arr) && all(isfinite, sde.neg.Vh_arr)
    @test r.Vh_arr[2] ≈ 8.0178e-02 rtol=1e-3
    @test abs(sde.neg.Vh_arr[2]) < 1e-3
    # the same slope on the pi phase: finite everywhere, same sign as the sigma one
    @test all(isfinite, r.Vh2_arr) && all(isfinite, sde.neg.Vh2_arr)
    @test r.Vh2_arr[2] ≈ 9.0152e-02 rtol=1e-3
    @test sign(r.Vh2_arr[2]) == sign(r.Vh_arr[2])
    @info "bank 2: V(+J)=$(r.V_arr[2])  V(-J)=$(sde.neg.V_arr[2])  slips=$(r.nslip_arr[2])"
end

@testset "a Julia checkpoint is read back" begin
    salida = mktempdir()
    cfg = cfg_fixture(out_dir=salida, mMax=3, iter_min=1, t_min_GL=0.0)
    sde = run_case(cfg; mode=Strict())
    p = sde.p
    s, jp, V, J = load_checkpoint(checkpoint_path(joinpath(salida, "restart"), "Jpos"),
                                    p, Strict())
    @test jp == 1
    @test all(s.G1 .=== sde.pos.estado.G1)
    @test all(s.hz .=== sde.pos.estado.hz)
    @test length(V) == 1 && length(J) == 1
end
