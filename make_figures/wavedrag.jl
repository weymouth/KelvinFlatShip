# drag stuff
include("../src/flatship.jl")
Cw(L,b) = quadgk(t->SpecialFunctions.besselj1(b*k(t))^2/(b^2*t^2*(1+2t^2))*(1-cos(L*hypot(1,t))),-Inf,0,Inf;atol=1e-6)[1]
Cw(5,1)

import AcceleratedKernels as AK
function Cw(L::AbstractArray,b::AbstractArray)  # multi-threaded evaluation of Cw
    z = Array{Float64}(undef,length(L),length(b))
    AK.foraxes(z,1) do i; for j in axes(z,2) 
        z[i,j] = Cw(L[i],b[j])
    end; end; z
end

# make plot
using Plots
L=range(1,6π,200); b=logrange(0.1,10,100)
z = Cw(L,b)
contourf(L,b,z',yscale=:log10,ylabel="b",levels=18,lw=0,
    xlabel="L",xticks=(π:π:6π,["π","2π","3π","4π","5π","6π"]))
savefig("wave_drag.png")