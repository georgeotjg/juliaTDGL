# The menu may manage a run it launched outside julia/runs; archives remain read-only.
using TDGLTUI, Test
const TUI = TDGLTUI
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
        @test TUI._managed_v4(personal)
        @test !TUI._managed_v4(archive)
        @test !row(personal)["on"]
        @test row(personal)["tag"] == "personal"
        write(joinpath(personal,"run.toml"),"# stopped before the first checkpoint\n")
        @test TUI._es_v4(personal)
        @test_throws ArgumentError TUI._v4_spawn(archive,"unused.toml","0")
        @test !isfile(joinpath(archive,".tdgl-tui-run"))
        @test !isfile(joinpath(archive,"launch.txt"))
        @test_throws ArgumentError TUI._v4_spawn(archive,"unused.toml","0";resume=true)
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
