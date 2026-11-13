using Test, TDGL
@testset "classic substep rotation near zero and pi" begin
    angles=Float64[0.0,pi,-pi,0.02,-0.02,0.4,-0.4]
    for t in (0.05,asin(0.05),pi-0.05,pi-asin(0.05)), s in (-1,1), d in (-1e-9,0.0,1e-9)
        push!(angles,s*t+d)
    end
    angles=sort(unique(angles)); ph=[a for scale in (1.0,1-2.6e-7,1+2.6e-7) for a in angles]
    scales=[r for r in (1.0,1-2.6e-7,1+2.6e-7) for a in angles]
    V=ones(ComplexF64,length(ph),1,1); U=reshape(scales.*cis.(-ph),size(V)); phase=angle.(V.*conj.(U))
    rng=CartesianIndices(V); large=[I for I in rng if abs(phase[I])>0.1]
    for n in (2,5,9)
        L=similar(V); R=similar(V); TDGL._pi_rot!(L,R,V,U,n,rng); reference=cis.(phase./n)
        @test L==V
        @test maximum(abs.(R.-reference))<4e-8
        @test maximum(abs(R[I]-reference[I]) for I in large)<2e-14
    end
end
