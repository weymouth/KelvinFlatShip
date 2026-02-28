# wave stuff
include("../src/flatship.jl")
ζ(x,y,b) = derivative(x->∫₂kelvin(x,y,-0.,b)*2/(π*b),x)

# make plot
using Plots
for b in (1,2.5)
    plot(xlabel="x",ylabel="y",size=(600,600));
    contour!(-20:0.1:1,-10:0.05:10,(x,y)->ζ(x,y,b)-ζ(x+5,y,b),levels=-8.5:1:8.5,clims=(-9,9))
    plot!(Shape([(0,b),(0,-b),(-5,-b),(-5,b)]),label="",c=:black,alpha=0.5,aspect_ratio=1)
    savefig("wave_field_b$(b).png")
end