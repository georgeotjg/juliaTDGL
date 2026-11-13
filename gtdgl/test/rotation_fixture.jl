# Manufactured temporal link rotations, shared by CPU and optional accelerator tests.
module RotationFixture
export inputs, maxmodulus
function inputs()
    angles=Float64[0.0,pi,-pi,0.02,-0.02,0.4,-0.4]
    for t in (0.05,asin(0.05),pi-0.05,pi-asin(0.05)), s in (-1,1), d in (-1e-9,0.0,1e-9)
        push!(angles,s*t+d)
    end
    angles=sort(unique(angles))
    ph=[a for scale in (1.0,1-2.6e-7,1+2.6e-7) for a in angles]
    scale=[r for r in (1.0,1-2.6e-7,1+2.6e-7) for a in angles]
    n=length(ph); push!(ph,ph[end]); push!(scale,scale[end])
    V=ones(ComplexF64,n+1,2,2)
    U=[scale[i]*cis(-ph[i]) for i in 1:n+1,j in 1:2,k in 1:2]
    omega=[i/64+j/32-k/16 for i in 1:n+1,j in 1:2,k in 1:2]
    phase=angle.(V.*conj.(U))
    ranges=(CartesianIndices((1:n,1:2,1:2)),CartesianIndices((1:n+1,1:1,1:2)),CartesianIndices((1:n+1,1:2,1:1)))
    (V=V,U=U,omega=omega,phase=phase,n=n,ranges=ranges,dt=1/32)
end
maxmodulus(a,rng)=maximum(abs(abs(a[I])-1) for I in rng)
end
