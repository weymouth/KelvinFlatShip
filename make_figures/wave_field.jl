# wave stuff
include("../src/flatship.jl")
∫₂kelvin(x,y,z,b=1) = ∫₂wavelike(x,y,z;b)+quadgk(y′->√(1-(y′/b)^2)*nearfield(x,y-y′,z),-b,b;atol=1e-4)[1]
ζ(x,y,z,b) = derivative(xp->∫₂kelvin(xp,y,z,b),x) * 2/(π*b)
import AcceleratedKernels as AK
function ζ(x::AbstractArray,y::AbstractArray,b)  # multi-threaded evaluation of ζ-field
    z = Array{Float64}(undef,length(x),length(y))
    AK.foraxes(z,1) do i; for j in axes(z,2) 
        z[i,j] = ζ(x[i],y[j],-0.,b)
    end; end; z
end

# make plot
using Plots
ζm = 8.5
for (b,h) in ((1,0.05),(2,0.1))
    x=-20:h:1; y=-10:h:10; z = clamp.((ζ(x,y,b)-ζ(x .+ 5,y,b)),-ζm,ζm)
    contourf(x,y,z',levels=18,clims=(-ζm,ζm),colormap=:balance,line=0)
    plot!(Shape([(0,b),(0,-b),(-5,-b),(-5,b)]),label="",c=:black,alpha=0.5,aspect_ratio=1,xlabel="x",ylabel="y",size=(600,600))
    savefig("wave_field_b$(b).png")
end

begin
    x = range(-10,1,2^10)
    plts = map(enumerate((0,0.6,1.2))) do (i,y)
        p = plot(ylabel="ζ(x,y=$y,z)",left_margin=5Plots.mm,yticks=-5:5:5)
        plot!(x,x->ζ(x,y,-0.1,1.),label="z=-0.1",c=colormap("Oranges",7)[4])
        plot!(x,x->ζ(x,y,-0.01,1.),label="z=-0.01",c=colormap("Oranges",7)[5])
        plot!(x,x->ζ(x,y,-0.,1.),label="z=0",c=colormap("Oranges",7)[6],lw=1.5)
        i==1 && plot!(xaxis=false,bottom_margin=-5Plots.mm)
        i==2 && plot!(xaxis=false,bottom_margin=-5Plots.mm,legend=false,top_margin=-5Plots.mm)
        i==3 && plot!(xlabel="x",top_margin=-5Plots.mm,legend=false)
        p
    end
    l = @layout [a{0.43h}; b{0.3h}; c{0.27h}]
    plot(plts..., layout=l, size=(400,600), link=:x)
    savefig("wavecutx.png")
end