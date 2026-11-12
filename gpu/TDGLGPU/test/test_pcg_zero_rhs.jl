using TDGL, TDGLGPU, KernelAbstractions, Test
const CUDA_ON = "cuda" in ARGS
CUDA_ON && @eval using CUDA
backend = CUDA_ON ? CUDABackend() : KernelAbstractions.CPU()
include(joinpath(pkgdir(TDGL), "test", "pcg_zero_fixture.jl"))

function make_device_solve(st)
    gs = GPUStepper(st, backend)
    function solve!(s)
        g = device_state(backend, s)
        TDGLGPU._pcg_up!(gs.sep, st.imp)
        TDGLGPU._implicit_curl!(g, gs.sep, gs)
        KernelAbstractions.synchronize(backend)
        host_state!(s, g)
        TDGLGPU._pcg_down!(st.imp, gs.sep)
        s
    end
end
@testset "device PCG zero source and checkpoint" begin
    pcg_zero_rhs_checks(make_device_solve;
        checkpoint_dir=joinpath(get(ENV, "TDGLGPU_TEST_WORKDIR", mktempdir()), "pcg_zero"))
end
println("PCG zero-source and checkpoint regression passed on ", CUDA_ON ? "CUDA" : "KA-CPU")
