# Repeated warmed step batches, with and without the cycle's host/device copies.
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Statistics, TOML, Pkg, SHA
BLAS.set_num_threads(1)
function cuda_backend()
    CUDA.functional() || error("CUDA is not functional; --cpu-backend selects KA-CPU explicitly")
    CUDA.CUDABackend()
end
function main(args)
    if isempty(args) || args[1] in ("-h", "--help")
        println("usage: benchmark.jl RUN.toml [--steps 20] [--repeats 3] [--cpu-backend] [--out out/gpu_benchmark]")
        println("Uses the configured initial seed and last current. Reports warmed batches; not an onset or whole-run benchmark.")
        return
    end
    opts=Dict{String,String}(); cpu=false; i=2
    while i<=length(args)
        k=args[i]
        if k=="--cpu-backend"
            cpu=true; i+=1
        elseif k in ("--steps","--repeats","--out")
            i<length(args) || error("$k requires a value")
            opts[k]=args[i+1]; i+=2
        else
            error("unknown option: $k")
        end
    end
    steps=parse(Int,get(opts,"--steps","20")); reps=parse(Int,get(opts,"--repeats","3"))
    steps>0 && reps>0 || error("steps and repeats must be positive")
    backend = if cpu
        KernelAbstractions.CPU()
    else
        @eval using CUDA
        Base.invokelatest(cuda_backend) # resolve the newly imported binding inside the latest world
    end
    # CUDA's optional methods must be visible after the import above.
    Base.invokelatest(measure, args, opts, cpu, steps, reps, backend)
end

function measure(args,opts,cpu,steps,reps,backend)
    out=get(opts,"--out",joinpath("out","gpu_benchmark")); mkpath(out)
    isfile(joinpath(out,"results.toml")) && error("$out already contains results; choose another output directory")
    c=load_config(args[1]); J=Float64(c.J[end]); H=c.H
    seed=TDGL.start_state(c,args[1])
    seed===nothing && (seed=initial_state(c.model,c.grid,c.masks,H;materials=get(c.scheme,:materials,nothing)))
    newst()=Stepper(c.model,c.grid,c.masks;c.scheme...)
    function cpu_batch(s,st)
        hj=selffield(st,J)
        for _ in 1:steps; step!(s,st;H=H,HJ=hj); end
    end
    samples=Dict{String,Any}[]
    # The first repetition compiles/warms the same methods but is not timed data.
    for rep in 0:reps
        a=deepcopy(seed); sa=newst()
        tc=@elapsed cpu_batch(a,sa)
        b=deepcopy(seed); sb=newst(); gs=GPUStepper(sb,backend); g=device_state(backend,b)
        hj=selffield(sb,J)
        tg=@elapsed gpu_steps!(g,gs,steps;H=H,HJ=hj)
        host_state!(b,g)
        fields=(:θx,:θy,:θz,:hx,:hy,:hz)
        aa=(a.G...,(getfield(a,f) for f in fields)...); bb=(b.G...,(getfield(b,f) for f in fields)...)
        err=maximum(maximum(abs.(x.-y))/max(maximum(abs,x),maximum(abs,y),1.0) for (x,y) in zip(aa,bb))
        err<=1e-6 || error("CPU/backend comparison failed: $err > 1e-6")
        s=deepcopy(seed); st=newst(); enable!(backend;fallback=:error)
        try
            TDGL.STEPS[](s,st,0;H=H,HJ=hj) # initialise cache, without advancing or consuming RNG
            tw=@elapsed TDGL.STEPS[](s,st,steps;H=H,HJ=hj)
            cc=(s.G...,(getfield(s,f) for f in fields)...)
            cycle_error=maximum(maximum(abs.(x.-y))/max(maximum(abs,x),maximum(abs,y),1.0) for (x,y) in zip(aa,cc))
            cycle_error<=1e-6 || error("CPU/cycle comparison failed: $cycle_error > 1e-6")
            if rep>0
                push!(samples,Dict("repeat"=>rep,"cpu_ms_per_step"=>tc*1000/steps,
                    "backend_ms_per_step"=>tg*1000/steps,"cycle_ms_per_step"=>tw*1000/steps,
                    "state_error"=>err,"cycle_state_error"=>cycle_error,
                    "cpu_fallback_steps"=>execution_stats().cpu_fallback_steps))
            end
        finally
            disable!()
        end
    end
    versions=Dict(string(p.name)=>string(p.version) for p in values(Pkg.dependencies()) if p.version!==nothing)
    result=Dict("julia"=>string(VERSION),"host_cpu"=>Sys.CPU_NAME,"backend"=>cpu ? "KA-CPU" : CUDA.name(CUDA.device()),
        "steps_per_batch"=>steps,"repeats"=>reps,"J"=>J,"H"=>H,"mesh"=>[c.grid.Nx,c.grid.Ny,c.grid.Nz],
        "config_sha256"=>bytes2hex(sha256(read(args[1]))),"versions"=>versions,"samples"=>samples,
        "benchmark_source_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "cpu_ms_median"=>median(s["cpu_ms_per_step"] for s in samples),
        "backend_ms_median"=>median(s["backend_ms_per_step"] for s in samples),
        "cycle_ms_median"=>median(s["cycle_ms_per_step"] for s in samples),
        "scope"=>"warmed batches; cycle includes transfers but excludes construction, measurements, protocol and checkpoints")
    open(io->TOML.print(io,result),joinpath(out,"results.toml"),"w")
    println("backend=",result["backend"],"; CPU/backend/cycle median ms/step: ",
        result["cpu_ms_median"]," / ",result["backend_ms_median"]," / ",result["cycle_ms_median"])
end
Base.invokelatest(main,ARGS)
