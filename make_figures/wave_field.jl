# wave stuff
include("../src/flatship.jl")
ζ(x,y,b) = derivative(x->∫₂kelvin(x,y,-0.,b)*2/(π*b),x)

# make plot
using Plots
h,ζm = 0.05,8.5
for b in (1,2)
    contourf(-20:h:1,-10:h:10,(x,y)->clamp(ζ(x,y,b)-ζ(x+5,y,b),-ζm,ζm),levels=18,clims=(-ζm,ζm),colormap=:balance,line=0)
    plot!(Shape([(0,b),(0,-b),(-5,-b),(-5,b)]),label="",c=:black,alpha=0.5,aspect_ratio=1,xlabel="x",ylabel="y",size=(600,600));
    savefig("wave_field_b$(b).png")
end