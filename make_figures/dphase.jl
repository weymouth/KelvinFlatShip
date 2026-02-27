# wave stuff
include("../src/pointsource.jl")
include("../src/flatship.jl")

using Plots
begin 
    x=-8; y = -x/4√2; Δg = logrange(0.1,50,5000)
    plt = plot(size=(500,300), xscale=:log10, yscale=:log10,ylims=(1e-12,1), xlabel="Δg", ylabel="|W error|", xlims=(0.1,100), xticks=[0.1,1,10,100])
    for n in 3:6
        z = -0.46416^n
        qgk = brutewavelike(x,y,z,atol=1e-12)
        plot!(Δg,Δg->abs(wavelike(x,y,z;Δg,ltol=-20)-qgk),label="z=$(round(z, digits=3))",c=colormap("Blues",6)[n])
    end
end;plt
savefig(plt,"dphase_z.png")

using FastGaussQuadrature
xlag2,wlag2 = gausslaguerre(2)
xlag8,wlag8 = gausslaguerre(8)
begin 
    z = -0.01; x=-8; y = -x/4√2
    qgk = brutewavelike(x,y,z,atol=1e-12)
    plt = plot(size=(500,300), xscale=:log10, yscale=:log10,ylims=(1e-12,1), xlabel="Δg", ylabel="|W error|", xlims=(0.1,100), xticks=[0.1,1,10,100])
    plot!(logrange(0.1,100,5000),Δg->abs(wavelike(x,y,z;Δg,ltol=-20,xlag=xlag2,wlag=wlag2)-qgk),label="Guass-Laguerre N=2",c=colormap("Blues",4)[2])
    plot!(logrange(0.1,50,5000),Δg->abs(wavelike(x,y,z;Δg,ltol=-20)-qgk),label="Guass-Laguerre N=4",c=colormap("Blues",4)[3])
    plot!(logrange(0.1,31,5000),Δg->abs(wavelike(x,y,z;Δg,ltol=-20,xlag=xlag8,wlag=wlag8)-qgk),label="Guass-Laguerre N=8",c=colormap("Blues",4)[4])
end;plt
savefig(plt,"dphase_GLn.png")
