# wave stuff
include("../src/flatship.jl")
∫₂kelvin(x,y,z,b=1) = ∫₂wavelike(x,y,z;b)+quadgk(y′->√(1-(y′/b)^2)*nearfield(x,y-y′,z),-b,b;atol=1e-4)[1]
ζb(x,y,z,b) = derivative(xp->∫₂kelvin(xp,y,z,b),x) * 2/(π*b)
import AcceleratedKernels as AK
function ζb(x::AbstractArray,y::AbstractArray,b)  # multi-threaded evaluation of ζ-field
    z = Array{Float64}(undef,length(x),length(y))
    AK.foraxes(z,1) do i; for j in axes(z,2)
        z[i,j] = ζb(x[i],y[j],-0.,b)
    end; end; z
end
ζp(x,y,z) = derivative(x->wavelike(x,y,z)+nearfield(x,y,z),x)

# make contour plots
using Plots
ζm = 10
for (b,h) in ((2,0.1),(0.5,0.05),(1,0.05))
    x=-20:h:1; y=-10:h:10; z = clamp.((ζb(x,y,b)-ζb(x .+ 5,y,b)),-ζm,ζm)
    contourf(x,y,z',levels=18,clims=(-ζm,ζm),colormap=:balance,line=0)
    plot!(Shape([(0,b),(0,-b),(-5,-b),(-5,b)]),label="",c=:black,alpha=0.5,
        aspect_ratio=1,xlabel="x",ylabel="y",size=(600,550),margin=0Plots.mm,
        xlims=(-20,1),ylims=(-10,10))
    savefig("wave_field_b$(b).png")
end

# make 3D surface plots
# Camera angle: (azimuth, elevation) where x goes right, z up, y back-left
begin
    b,h = 1,0.05
    x=-20:h:1; y=-10:h:10; z = ζb(x,y,b) .- ζb(x .+ 5,y,b)
    surface(x,y,0.1z',colormap=:balance,
            camera=(-10,10),
            xlabel="x",ylabel="y",
            size=(1800,1600),
            zlims=(-ζm,ζm))
    savefig("flatship_surface.png")

    z = ζp.(x,y',-0.1)
    surface(x,y,0.1z',colormap=:balance,
            camera=(-10,10),
            xlabel="x",ylabel="y",
            size=(1800,1600),
            zlims=(-ζm,ζm))
    savefig("pointsource_surface.png")

end