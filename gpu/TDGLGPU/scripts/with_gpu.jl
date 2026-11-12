# Any script of release 2 (tdgl2/scripts: refine.jl, hysteresis.jl, mh.jl, run.jl, ...) with the step on a GPU (TDGLGPU):
#   julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/with_gpu.jl [--cpu-backend] [--require-gpu] tdgl2/scripts/SCRIPT.jl ARGS...
# Every ladder the script walks steps its cycles on the device (TDGLGPU.enable!); the script itself is not changed.
using TDGL, TDGLGPU, KernelAbstractions
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    println("usage: julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/with_gpu.jl [--cpu-backend] [--require-gpu] SCRIPT.jl ARGS...")
    exit(0)
end
function wrapper_options!(args)
    cpu_backend = false; require_gpu = false
    while !isempty(args) && startswith(args[1], "--")
        flag = popfirst!(args)
        flag == "--cpu-backend" ? (cpu_backend = true) : flag == "--require-gpu" ? (require_gpu = true) :
            error("unknown wrapper option: $flag")
    end
    cpu_backend, require_gpu
end
cpu_backend, require_gpu = wrapper_options!(ARGS)
cpu_backend && require_gpu && error("--require-gpu cannot be combined with --cpu-backend")
isempty(ARGS) && error("missing script")
if cpu_backend
    TDGLGPU.enable!(KernelAbstractions.CPU(); fallback=:cpu)
else
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional; use --cpu-backend for an explicit KA-CPU check")
    TDGLGPU.enable!(CUDABackend(); fallback=require_gpu ? :error : :cpu)
end
script = popfirst!(ARGS)
const _reported = Ref(false)
function report_stats()
    _reported[] && return
    _reported[] = true
    stats = TDGLGPU.execution_stats()
    println("backend=", stats.backend, ", backend_steps=", stats.backend_steps,
            ", cpu_fallback_steps=", stats.cpu_fallback_steps)
end
atexit(report_stats) # a wrapped CLI may call exit() before returning from include
try
    include(abspath(script))
finally
    report_stats()
    TDGLGPU.disable!()
end
