# An early stop, or an unstarted polarity, must keep the configured initial state.
using TDGL, Test
@testset "resume before a checkpoint keeps the configured seed" begin
    mktempdir() do root
        cfg = joinpath(root,"input.toml")
        write(cfg,"""
        [material]
        file = "mgb2_model_II.toml"
        kappa = 2.0
        [mesh]
        nx = 12
        ny = 8
        nz = 4
        dx = 0.5
        [geometry]
        GEOM_MODE = "WEDGE"
        inj_margin = 2
        [scheme]
        threads = false
        [protocol]
        t_min_GL = 0.1
        t_min_GL_res = 0.2
        mMax = 12
        lMax = 2
        iter_min = 4
        delta_v = 2
        V_win = 4
        V_win_min_res = 6
        [run]
        J = [0.0, 0.2]
        H = 0.03
        start = "seed.jld2"
        """)
        c=load_config(cfg); s=initial_state(c.model,c.grid,c.masks,c.H)
        for G in s.G; G .*= 0.3 + 0.2im; end
        save_state(joinpath(root,"seed.jld2"),s)
        reference=run_config(cfg;out_dir=joinpath(root,"full"))
        early=joinpath(root,"early"); mkpath(early)
        stop=joinpath(early,"STOP.txt"); write(stop,"test early stop\n")
        run_config(cfg;out_dir=early)
        @test !isfile(joinpath(early,"restart","state_J+.jld2"))
        mv(stop,joinpath(root,"stopped_marker.txt"))
        resumed=run_config(joinpath(early,"run.toml");out_dir=early,resume=true)
        @test resumed.pos.state.G == reference.pos.state.G
        @test resumed.neg.state.G == reference.neg.state.G
        partial=joinpath(root,"partial")
        run_config(cfg;out_dir=partial,polarity=:pos)
        completed=run_config(joinpath(partial,"run.toml");out_dir=partial,resume=true)
        @test completed.pos.state.G == reference.pos.state.G
        @test completed.neg.state.G == reference.neg.state.G
    end
end
