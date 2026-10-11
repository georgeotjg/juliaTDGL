@testset "dt_scale and checkpoints with temperature and residual" begin
    kw = (; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
          kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30, gamma12=0.001, H_const=0.5, J_list=[0.0],
          selftest=false, snap_every=0, verbose=false, use_threads=false)
    # dt_scale: off = the same step to the bit; on = that step times the factor
    p1 = derive_params(Config(; kw..., out_dir=""))
    @test derive_params(Config(; kw..., out_dir="", dt_scale=1.0)).dt === p1.dt
    @test derive_params(Config(; kw..., out_dir="", dt_scale=0.5)).dt == 0.5 * p1.dt
    @test_throws ErrorException derive_params(Config(; kw..., out_dir="", dt_scale=0.0))
    @test_throws ErrorException derive_params(Config(; kw..., out_dir="", dt_scale=1.5))
    # theta and Q survive a checkpoint
    dir = mktempdir()
    cfg = Config(; kw..., out_dir=dir, use_heat=true, use_phi=true, tau_Q=1.0)
    p = derive_params(cfg); g = build_geometry(cfg, p); s = initial_state(p, g, 0.5, Strict())
    p.Temp .= 0.01 .* reshape(sin.(1:length(p.Temp)), size(p.Temp))
    Q = 0.02 .* reshape(cos.(1:length(p.Q_restart)), size(p.Q_restart))
    TDGL.save_checkpoint(p, s, "J+", 3, [0.1, 0.2, 0.3], [0.0, 0.1, 0.2]; Qimb=Q)
    T0 = copy(p.Temp); fill!(p.Temp, 0.0)
    s2, jp, V, J = load_checkpoint(joinpath(dir, "restart", "state_Jpos.mat"), p, Strict())
    @test jp == 3 && V == [0.1, 0.2, 0.3]
    @test p.Temp == T0
    @test p.Q_restart == Q
    @test s2.G1 == s.G1
    # without heat or residual the checkpoint has neither field
    cfg0 = Config(; kw..., out_dir=dir)
    p0 = derive_params(cfg0); s0 = initial_state(p0, build_geometry(cfg0, p0), 0.5, Strict())
    w = read_state(TDGL.save_checkpoint(p0, s0, "J-", 1, [0.0], [0.0]))
    @test !haskey(w, "Temp") && !haskey(w, "Qimb")
end
