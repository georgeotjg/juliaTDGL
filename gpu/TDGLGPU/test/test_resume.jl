# A closed-rung checkpoint resumed on a fresh backend must reproduce continuation.
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Test, TOML
BLAS.set_num_threads(1)
const CUDA_ON = "cuda" in ARGS
CUDA_ON && @eval using CUDA
backend = CUDA_ON ? CUDABackend() : KernelAbstractions.CPU()
root = joinpath(get(ENV, "TDGLGPU_TEST_WORKDIR", mktempdir()), "resume")
mkpath(root)
md = two_band(xi_ratio=4.0, κ=2.0)
gr = Grid(12, 8, 4; dx=0.5, buf=2)
mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=2), gr)
pr = Protocol(t_min_GL=0.1, t_min_GL_res=0.2, mMax=14, lMax=2,
              iter_min=4, delta_v=2, V_win=4, V_win_min_res=6, stop_after_V=0)
Js = [0.0, 0.2, 0.4]
results = Dict{String,Any}[]
function checkpoint_csv_close(a,b;tol=1e-7)
    rows(v)=[Base.split(l,',') for l in v if !startswith(l,"#")]
    ra,rb=rows(a),rows(b)
    closecell(x,y)=x==y || let u=tryparse(Float64,x),v=tryparse(Float64,y)
        u!==nothing && v!==nothing && (isequal(u,v) || isapprox(u,v;atol=tol,rtol=tol))
    end
    length(ra)==length(rb) && all(length(x)==length(y) && all(closecell(u,v) for (u,v) in zip(x,y)) for (x,y) in zip(ra,rb))
end
@testset "backend checkpoint continuation" begin
    for (bc, physics) in ((:bar, :plain), (:slab, :plain), (:open, :plain), (:bar, :noise_kwt), (:open, :heat))
        @testset "$bc/$physics with insulating PCG" begin
            model = physics === :noise_kwt ? two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=2.0) : md
            extra = physics === :noise_kwt ? (; noise=1e-5, noise_seed=43) :
                    physics === :heat ? (; heat=Heat(joule=true, margin=2)) : (;)
            newstepper() = Stepper(model, gr, mk; threads=false, σvac=0.01, selffield_bc=bc, extra...)
            full = joinpath(root, "$(bc)_$(physics)", "full")
            splitdir = joinpath(root, "$(bc)_$(physics)", "split")
            st0 = newstepper()
            TDGLGPU.enable!(backend; fallback=:error)
            try
                a = ladder(st0, initial_state(model, gr, mk, 0.03), Js; H=0.03, pr=pr, out_dir=full)
                TDGLGPU.enable!(backend; fallback=:error)
                ladder(newstepper(), initial_state(model, gr, mk, 0.03), Js[1:2]; H=0.03, pr=pr, out_dir=splitdir)
                restored = newstepper()
                TDGLGPU.enable!(backend; fallback=:error)
                b = resume_ladder(restored, Js; H=0.03, pr=pr, out_dir=splitdir)
                @test (CUDA_ON ? isapprox(a.V,b.V;atol=1e-7,rtol=1e-7) : a.V==b.V) && a.conv == b.conv && a.m == b.m
                fields = (:θx,:θy,:θz,:φx,:φy,:φz,:hx,:hy,:hz)
                arrays_a = (a.state.G..., (getfield(a.state,f) for f in fields)...)
                arrays_b = (b.state.G..., (getfield(b.state,f) for f in fields)...)
                state_error = maximum(maximum(abs.(x.-y)) / max(maximum(abs,x),maximum(abs,y),1.0) for (x,y) in zip(arrays_a,arrays_b))
                tolerance = CUDA_ON ? 1e-7 : physics === :noise_kwt ? 1e-8 : 0.0
                if tolerance > 0
                    # Checkpoints reconstruct cis(-θ) on the host. Device rounding can
                    # change PCG termination; seeded draws and verdicts must still agree.
                    @test state_error <= tolerance
                    @test maximum(maximum(abs.(x.-y)) for (x,y) in zip(st0.imp.x,restored.imp.x)) <= tolerance
                else
                    @test a.state.G == b.state.G
                    @test all(getfield(a.state, f) == getfield(b.state, f) for f in fields)
                    @test TDGL._pcg_snap(st0) == TDGL._pcg_snap(restored)
                end
                # The checkpoint holds the boundary clock and ramp, not FFT scratch.
                @test st0.open === nothing || (st0.open.count == restored.open.count &&
                    all(CUDA_ON ? isapprox(x,y;atol=1e-7,rtol=1e-7) : x==y for (x,y) in
                        zip((st0.open.old...,st0.open.new...),(restored.open.old...,restored.open.new...))))
                csv_a=readlines(joinpath(full,"output","rungs_J+.csv"))
                csv_b=readlines(joinpath(splitdir,"output","rungs_J+.csv"))
                @test checkpoint_csv_close(csv_a,csv_b)
                CUDA_ON || @test csv_a == csv_b
                @test execution_stats().cpu_fallback_steps == 0
                st0.noise === nothing || @test TDGL._noise_snap(st0) == TDGL._noise_snap(restored)
                st0.heat === nothing || @test CUDA_ON ? isapprox(st0.heat.θ,restored.heat.θ;atol=1e-7,rtol=1e-7) : st0.heat.θ == restored.heat.θ
                if bc === :open
                    # Cross a convolution update: the omitted scratch must be rebuilt.
                    n = st0.open.every + 1
                    hj = selffield(st0, Js[end])
                    TDGLGPU.enable!(backend; fallback=:error)
                    TDGL.STEPS[](a.state, st0, n; H=0.03, HJ=hj)
                    TDGLGPU.enable!(backend; fallback=:error)
                    TDGL.STEPS[](b.state, restored, n; H=0.03, HJ=hj)
                    @test CUDA_ON ? all(isapprox(x,y;atol=1e-7,rtol=1e-7) for (x,y) in zip(a.state.G,b.state.G)) : a.state.G == b.state.G
                    @test all(CUDA_ON ? isapprox(getfield(a.state,f),getfield(b.state,f);atol=1e-7,rtol=1e-7) :
                                       getfield(a.state,f)==getfield(b.state,f) for f in fields)
                    if CUDA_ON
                        @test st0.open.count == restored.open.count &&
                              all(isapprox(x,y;atol=1e-7,rtol=1e-7) for (x,y) in zip((st0.open.old...,st0.open.new...),
                                                                                 (restored.open.old...,restored.open.new...)))
                    else
                        @test open_extra(st0) == open_extra(restored)
                    end
                end
                push!(results, Dict("boundary"=>string(bc), "physics"=>string(physics), "pcg"=>true,
                                    "state_error"=>state_error,"state_tolerance"=>tolerance,
                                    "fresh_stepper_resume_exact"=>state_error == 0.0,
                                    "backend"=>execution_stats().backend,"passed"=>true))
            finally
                TDGLGPU.disable!()
            end
        end
    end
end
open(io -> TOML.print(io, Dict("cases"=>results)), joinpath(root,"results.toml"), "w")
