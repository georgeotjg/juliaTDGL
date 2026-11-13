# ---------------------------------------------------------------------------
#  Resuming a run, through run_case as a user does it, for every kind of state the checkpoint carries
#  (base, :prod with its previous-step links, heat θ with Joule, Φ with a relaxing residual Q, drag,
#  physical temperature, a shape file), in both checkpoint formats. A ladder stopped after two points
#  and resumed for the third must end in the state of the same ladder run without a stop, to the bit,
#  and the JLD2 and Octave-text checkpoints must give the same resumed run.
# ---------------------------------------------------------------------------
@testset "resume from a checkpoint: every state, both formats" begin
    base = (; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0,
            kappa_p=2.0, xi_ratio=1.30, gamma12=0.001, H_const=0.50, t_min_GL=0.0, t_min_GL_res=0.0,
            mMax=60, iter_min=5, eps_conv=1e-14, selftest=false, snap_every=0, vtk_every=0,
            verbose=false, use_threads=false)
    forma = joinpath(@__DIR__, "..", "shapes", "cat_WEDGE.toml")
    casos = [("base", :legacy, (;)),
             ("prod", :prod, (;)),
             ("heat+Joule", :legacy, (; use_heat=true, gamma_bath=0.04, use_joule=true, heat_margin=0)),
             ("Phi+tau_Q", :legacy, (; use_phi=true, tau_Q=1.0)),
             ("drag", :legacy, (; use_drag=true, nu_drag=0.05)),
             ("T_physical", :legacy, (; T_physical=true, T_red=0.3)),
             ("shape file", :legacy, (; GEOM_MODE="CUSTOM", geom_file=forma))]
    ignora = ("created_by", "tdgl_version")
    igual(a, b) = all(k -> k in ignora || (haskey(b, k) && all(isequal.(a[k], b[k]))), keys(a)) &&
                  all(k -> k in ignora || haskey(a, k), keys(b))
    for (nombre, esquema, extra) in casos
        final = Dict{String,Any}()
        for fmt in ("jld2", "octave")
            mk(J; kw...) = Config(esquema; base..., extra..., J_list=J, checkpoint_format=fmt, kw...)
            cont = mktempdir(); res = mktempdir()
            run_case(mk([0.05, 0.10]; out_dir=cont))                                   # without a stop
            run_case(mk([0.05]; out_dir=res))                                          # two points (0, 0.05)
            run_case(mk([0.05, 0.10]; out_dir=res, restart_pos=true, restart_neg=true)) # the third, resumed
            for pol in ("Jpos", "Jneg")
                a = read_state(checkpoint_path(joinpath(cont, "restart"), pol))
                b = read_state(checkpoint_path(joinpath(res, "restart"), pol))
                @test endswith(checkpoint_path(joinpath(res, "restart"), pol), fmt == "jld2" ? ".jld2" : ".mat")
                ok = igual(a, b)
                ok || @info "resume differs from the uninterrupted run" nombre fmt pol
                @test ok
                final[fmt*pol] = b
            end
        end
        for pol in ("Jpos", "Jneg")
            @test igual(final["jld2"*pol], final["octave"*pol])
        end
    end
end
