using Test, TDGL
include("rotation_fixture.jl")
using .RotationFixture: inputs, maxmodulus

@testset "temporal link rotations near zero and pi" begin
    f=inputs(); V,U,phase=f.V,f.U,f.phase
    rng=CartesianIndices(V); large=[I for I in rng if abs(phase[I])>0.1]
    for c in (0.25,0.5,0.9,1.0)
        L=similar(V); TDGL._rotfrac!(L,V,U,c,rng); reference=V.*cis.(c.*phase)
        @test maximum(abs.(L.-reference))<4e-7
        @test maximum(abs(L[I]-reference[I]) for I in large)<2e-14
        @test maxmodulus(L,rng)<3e-7
    end
    for n in (2,5,9)
        L=similar(V); R=similar(V); TDGL._rot!(L,R,V,U,n,rng)
        reference=cis.(phase./n)
        @test L==V
        @test maximum(abs.(R.-reference))<4e-8
        @test maximum(abs(R[I]-reference[I]) for I in large)<2e-14
    end
    for (axis,rng) in enumerate(f.ranges)
        offset=CartesianIndex(ntuple(i->i==axis ? 1 : 0,3)); R=zeros(size(V))
        TDGL._link_rate!(R,V,U,f.omega,f.dt,rng,offset)
        expected(I)=phase[I]+(f.omega[I+offset]-f.omega[I])*f.dt
        @test maximum(abs(R[I]-expected(I)) for I in rng)<4e-8
        @test maximum(abs(R[I]-expected(I)) for I in rng if abs(phase[I])>0.1)<2e-14
    end
    @test abs(TDGL._cis(0.05)-cis(0.05))<2.605e-7
end
