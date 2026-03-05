# wave drag
include("../src/flatship.jl")
Cw(L,b) = quadgk(t->SpecialFunctions.besselj1(b*k(t))^2/(b^2*t^2*(1+2t^2))*(1-cos(L*hypot(1,t))),-Inf,0,Inf;atol=1e-6)[1]

# multi-threaded evaluation
import AcceleratedKernels as AK
function Cw(L::AbstractArray,b::AbstractArray)
    z = Array{Float64}(undef,length(L),length(b))
    AK.foraxes(z,1) do i; for j in axes(z,2) 
        z[i,j] = Cw(L[i],b[j])
    end; end; z
end

# make plot
using Plots
L=logrange(π/10,10π,200); b=logrange(0.1,10,100)
z = Cw(L,b)
contourf(L,b,z',levels=18,lw=0,cmap=:amp,
    ylabel="b",yscale=:log10,
    xlabel="L",xscale=:log10,xticks=([π/10,π,10π],["π/10","π","10π"]))
savefig("wave_drag.png")