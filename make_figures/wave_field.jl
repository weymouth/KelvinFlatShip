# wave stuff
include("../src/flatship.jl")
∫₂kelvin(x,y,z,b=1) = ∫₂wavelike(x,y,z;b)+quadgk(y′->√(1-(y′/b)^2)*nearfield(x,y-y′,z),-b,b;atol=1e-4)[1]
import AcceleratedKernels as AK
function ζ(x,y,b)  # multi-threaded evaluation of ζ
    z = Array{Float64}(undef,length(x),length(y))
    AK.foraxes(z,1) do i; for j in axes(z,2) 
        z[i,j] = derivative(x->∫₂kelvin(x,y[j],-0.,b),x[i])
    end; end; z
end

# make plot
using Plots
ζm = 8.5
for (b,h) in ((1,0.05),(2,0.1))
    x=-20:h:1; y=-10:h:10; z = clamp.((ζ(x,y,b)-ζ(x .+ 5,y,b)) .* 2/(π*b),-ζm,ζm)
    contourf(x,y,z',levels=18,clims=(-ζm,ζm),colormap=:balance,line=0)
    plot!(Shape([(0,b),(0,-b),(-5,-b),(-5,b)]),label="",c=:black,alpha=0.5,aspect_ratio=1,xlabel="x",ylabel="y",size=(600,600))
    savefig("wave_field_b$(b).png")
end