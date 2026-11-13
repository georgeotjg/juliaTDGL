# The menu may manage a run it launched outside tdgl/runs; archives remain read-only.
using TDGLTUI, Test
const TUI = TDGLTUI
@testset "software and run discovery roots" begin
    expected = isdefined(TUI, :COLAB_SH) ? dirname(realpath(TUI.SOFTWARE)) : realpath(TUI.SOFTWARE)
    @test TUI.BASE == expected
    @test isfile(TUI.EDITOR)
    @test isfile(TUI.GTDGL_RUN)
    @test basename(TUI.CLASSIC) == "tdgl"
    @test basename(TUI.GTDGL) == "gtdgl"
    if !isempty(TUI.WORKSPACE.relocated)
        for (old, current) in TUI.WORKSPACE.relocated
            @test TUI._realpath(old) == realpath(current)
            @test TUI._realpath(joinpath(old, "runs")) == realpath(joinpath(current, "runs"))
        end
    end
end
@testset "any-band window results" begin
    mktempdir() do dir
        write(joinpath(dir, "run_info.txt"), "t_cycle=1.0\n")
        write(joinpath(dir, "j_history.csv"),
            "time,pol,J,V,conv,it,Nv_s,Nv_p,psi_p2,onset_class\n" *
            "now,J+,0.1,0,F,20,0,0,1,S\n" *
            "now,J+,0.2,0,*,20,0,0,1,U\n" *
            "now,J+,0.3,4,V,20,0,0,1,D\n")
        text = join(TUI.quick_result(dir), "\n")
        @test occursin("Ic+ ∈ (0.1, 0.3)", text)
        @test occursin("(any band)", text) && occursin("undecided: +0.2", text)
    end
end
@testset "managed run and archive distinction" begin
    mktempdir() do work
        personal = joinpath(work, "personal"); archive = joinpath(work, "archive")
        default = joinpath(work, "default")
        for d in (personal, archive, default)
            mkpath(d); write(joinpath(d, "STATUS.txt"), "test done\n")
        end
        write(joinpath(personal, ".tdgl-tui-run"), "release 2 run launched by the terminal menu\n")
        empty!(TUI.ROOTS); append!(TUI.ROOTS, (default, personal, archive))
        rows = TUI.collect_runs()
        row(d) = only(filter(r -> r["dir"] == d, rows))
        @test !row(default)["frozen"]
        @test !row(personal)["frozen"]
        @test row(archive)["frozen"]
        @test TUI._managed_gtdgl(personal)
        @test !TUI._managed_gtdgl(archive)
        @test !row(personal)["on"]
        @test row(personal)["tag"] == "personal"
        alias = joinpath(work, "default_alias")
        symlink(default, alias; dir_target=true)
        push!(TUI.ROOTS, alias)
        @test count(r -> TUI._realpath(r["dir"]) == default, TUI.collect_runs()) == 1
        @test TUI._dentro(joinpath(alias, "STATUS.txt"), default)
        write(joinpath(personal,"run.toml"),"# stopped before the first checkpoint\n")
        @test TUI._es_gtdgl(personal)
        @test_throws ArgumentError TUI._gtdgl_spawn(archive,"unused.toml","0")
        @test !isfile(joinpath(archive,".tdgl-tui-run"))
        @test !isfile(joinpath(archive,"launch.txt"))
        @test_throws ArgumentError TUI._gtdgl_spawn(archive,"unused.toml","0";resume=true)
        # Resolve XDG at runtime, rather than caching the build machine's directory.
        withenv("XDG_CONFIG_HOME" => joinpath(work, "config_a")) do
            TUI._cambiar_carpeta(personal, false)
            @test TUI._carpetas_guardadas() == [personal]
        end
        withenv("XDG_CONFIG_HOME" => joinpath(work, "config_b")) do
            @test isempty(TUI._carpetas_guardadas())
            TUI._cambiar_carpeta(archive, false)
            @test TUI._carpetas_guardadas() == [archive]
        end
    end
end
