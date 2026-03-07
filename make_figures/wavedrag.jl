# wave drag
include("../src/flatship.jl")
Cw(L,b) = 2π*quadgk(t->SpecialFunctions.besselj1(b*k(t))^2/(b*k(t))^2*kₓ(t)*(1-cos(L*hypot(1,t))),0,Inf;atol=1e-6)[1]

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
L=range(π/10,6π,200); b=logrange(0.1,10,100)
z = Cw(L,b)
contourf(L,b,z',levels=18,lw=0,cmap=:amp,
    ylabel="      b",yscale=:log10,yguidefontrotation=-90,
    xlabel="L",xticks=([π,3π,5π],["π","3π","5π"]),
    size=(600,300),colorbar_title="\nCw/q₀²",rightmargin=5Plots.mm)
savefig("wave_drag.png")