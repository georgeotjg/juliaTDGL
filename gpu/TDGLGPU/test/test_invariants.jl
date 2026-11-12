# Analytic checks of the accelerator, independent of CPU-versus-GPU agreement.
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Test, TOML
BLAS.set_num_threads(1)
const CUDA_ON = "cuda" in ARGS
CUDA_ON && @eval using CUDA
backend = CUDA_ON ? CUDABackend() : KernelAbstractions.CPU()

@testset "analytic uniform solutions on the accelerator" begin
    for gamma in (0.0, 10.0)
        md = Model((Band(Γ=2.0, γkwt=gamma),), zeros(1, 1); κ=2.0)
        gr = Grid(12, 8, 6; dx=0.5)
        st = Stepper(md, gr, box_masks(gr); threads=false, rkl=false)
        s = initial_state(md, gr, st.masks, 0.0)
        gs = GPUStepper(st, backend)
        g = device_state(backend, s)
        gpu_steps!(g, gs, 8; H=0.0, HJ=0.0)
        host_state!(s, g)
        interior = (gr.x1+1:gr.x2, gr.y1+1:gr.y2, gr.z1+1:gr.z2)
        @test maximum(abs.(view(s.G[1], interior...) .- 1)) < 1e-13
        @test maximum(maximum(abs, h) for h in (s.hx, s.hy, s.hz)) < 1e-13

        # A spatially uniform real perturbation has no current or diffusion. Its
        # exact Euler update is the local GL/KWT scalar equation, including Γ.
        s = initial_state(md, gr, st.masks, 0.0)
        p = 1.0 + 1e-5; s.G[1] .*= p
        expected = p + st.dt / 2 * (1 - p^2) * p / sqrt(1 + gamma^2 * p^2)
        g = device_state(backend, s)
        gpu_steps!(g, gs, 1; H=0.0, HJ=0.0)
        host_state!(s, g)
        @test maximum(abs.(view(s.G[1], interior...) .- expected)) < 1e-13
    end
end

@testset "a pure lattice gauge is stationary" begin
    md = Model((Band(),), zeros(1, 1); κ=2.0)
    gr = Grid(12, 8, 6; dx=0.5)
    st = Stepper(md, gr, box_masks(gr); threads=false, rkl=false)
    s = initial_state(md, gr, st.masks, 0.0)
    chi = zeros(size(s.G[1]))
    for k in gr.z1+2:gr.z2-1, j in gr.y1+2:gr.y2-1, i in gr.x1+2:gr.x2-1
        chi[i,j,k] = 0.3sin(0.4i + 0.7j - 0.2k)
    end
    phase = cis.(chi)
    s.G[1] .*= phase; s.F[1] .*= phase
    for (theta, prev, U, oldU, axis) in zip((s.θx, s.θy, s.θz), (s.φx, s.φy, s.φz),
                                            (s.Vx, s.Vy, s.Vz), (s.Ux, s.Uy, s.Uz), 1:3)
        delta = diff(chi; dims=axis)
        theta .+= delta; prev .+= delta
        U .*= cis.(-delta); oldU .*= cis.(-delta)
    end
    expected = copy(s.G[1])
    g = device_state(backend, s); gs = GPUStepper(st, backend)
    gpu_steps!(g, gs, 6; H=0.0, HJ=0.0)
    host_state!(s, g)
    @test maximum(abs.(s.G[1] .- expected)) < 1e-12
    @test maximum(maximum(abs, h) for h in (s.hx, s.hy, s.hz)) < 1e-12
end

println("Analytic checks passed: equilibrium, radial GL/KWT relaxation and stationary pure gauge.")
