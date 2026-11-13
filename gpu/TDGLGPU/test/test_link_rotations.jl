# Actual fused/unfused kernels, with manufactured temporal rotations and CPU references.
using TDGL, TDGLGPU, KernelAbstractions, Test, TOML, SHA
const CUDA_ON = "cuda" in ARGS
CUDA_ON && @eval using CUDA
backend = CUDA_ON ? CUDABackend() : KernelAbstractions.CPU()
include(joinpath(pkgdir(TDGL),"test","rotation_fixture.jl"))
using .RotationFixture: inputs
const METRICS=Dict{String,Any}[]

function check_values(actual,reference,rng,kind)
    a=Array(actual)
    err=maximum(abs(a[I]-reference[I]) for I in rng)
    @test err<1e-13 # a different branch near the cutoff differs by ~1e-7, not a few ulps
    push!(METRICS,Dict("kernel"=>kind,"max_cpu_difference"=>err))
end
function launch(kernel,nd,args...)
    kernel(backend,(32,1,1))(args...;ndrange=nd)
    KernelAbstractions.synchronize(backend)
end

@testset "device temporal rotations: fused/unfused boundary parity" begin
    f=inputs(); sz=size(f.V); allrng=CartesianIndices(f.V); origin=(0,0,0)
    V=TDGLGPU._dev(backend,f.V); U=TDGLGPU._dev(backend,f.U); omega=TDGLGPU._dev(backend,f.omega)
    dims=(f.n+1,2,2); bounds=(1,f.n,1,1,1,1)
    function blank(T)
        TDGLGPU._dev(backend,zeros(T,sz))
    end
    for c in (0.25,0.5,0.9,1.0)
        cpu=similar(f.V); TDGL._rotfrac!(cpu,f.V,f.U,c,allrng)
        reference=f.V.*cis.(c.*f.phase)
        @test maximum(abs(cpu[I]-reference[I]) for I in allrng if abs(f.phase[I])>0.1)<2e-14
        L=blank(ComplexF64)
        launch(TDGLGPU._rotfrac_k!,sz,L,V,U,c,origin)
        check_values(L,cpu,allrng,"fraction_single")
        out=ntuple(_->blank(ComplexF64),3)
        launch(TDGLGPU._rotfrac3_k!,dims,out...,V,V,V,U,U,U,c,bounds...)
        for axis in 1:3; check_values(out[axis],cpu,f.ranges[axis],"fraction_fused_$axis"); end
    end
    for n in (2,5,9)
        cpu=similar(f.V); root=similar(f.V); TDGL._rot!(cpu,root,f.V,f.U,n,allrng)
        @test maximum(abs(root[I]-cis(f.phase[I]/n)) for I in allrng if abs(f.phase[I])>0.1)<2e-14
        L=blank(ComplexF64); R=blank(ComplexF64)
        launch(TDGLGPU._rot_k!,sz,L,R,V,U,n,origin)
        check_values(L,cpu,allrng,"root_source_single"); check_values(R,root,allrng,"root_single")
        out=ntuple(_->blank(ComplexF64),3); roots=ntuple(_->blank(ComplexF64),3)
        launch(TDGLGPU._rot3_k!,dims,out...,roots...,V,V,V,U,U,U,n,bounds...)
        for axis in 1:3
            check_values(out[axis],cpu,f.ranges[axis],"root_source_fused_$axis")
            check_values(roots[axis],root,f.ranges[axis],"root_fused_$axis")
        end
    end
    rates=ntuple(_->zeros(Float64,sz),3); out=ntuple(_->blank(Float64),3)
    for axis in 1:3
        offset=CartesianIndex(ntuple(i->i==axis ? 1 : 0,3)); rng=f.ranges[axis]
        TDGL._link_rate!(rates[axis],f.V,f.U,f.omega,f.dt,rng,offset)
        expected(I)=f.phase[I]+(f.omega[I+offset]-f.omega[I])*f.dt
        @test maximum(abs(rates[axis][I]-expected(I)) for I in rng if abs(f.phase[I])>0.1)<2e-14
        nd=size(rng)
        launch(TDGLGPU._lrate_k!,nd,out[axis],V,U,omega,f.dt,origin,Tuple(offset))
        check_values(out[axis],rates[axis],rng,"rate_single_$axis")
    end
    out=ntuple(_->blank(Float64),3)
    launch(TDGLGPU._lrate3_k!,dims,out...,V,V,V,U,U,U,omega,f.dt,bounds...)
    for axis in 1:3; check_values(out[axis],rates[axis],f.ranges[axis],"rate_fused_$axis"); end
end

work=joinpath(get(ENV,"TDGLGPU_TEST_WORKDIR",joinpath(pwd(),"out","gpu_validation")),"link_rotations")
mkpath(work)
result=Dict("passed"=>true,"backend"=>CUDA_ON ? "CUDA" : "KA-CPU","manufactured_cases"=>inputs().n,
            "max_cpu_difference"=>maximum(r["max_cpu_difference"] for r in METRICS),"checks"=>METRICS,
            "cpu_source_sha256"=>bytes2hex(sha256(read(joinpath(pkgdir(TDGL),"src","bands.jl")))),
            "gpu_source_sha256"=>bytes2hex(sha256(read(joinpath(pkgdir(TDGLGPU),"src","TDGLGPU.jl")))))
open(io->TOML.print(io,result),joinpath(work,"results.toml"),"w")
println("LINK ROTATION PASS: ",result["backend"],", ",result["manufactured_cases"]," inputs; max CPU difference ",result["max_cpu_difference"])
