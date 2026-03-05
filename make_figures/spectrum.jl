# wave stuff
include("../src/pointsource.jl")
∂ₓW(x,y,z) = derivative(x->wavelike(x,y,z),x)
include("../src/flatship.jl")
∂ₓWᵦ(x,y,z) = derivative(x->∫₂wavelike(x,y,z)*2/π,x)

# spectrum stuff
using DSP
spectrum(f) = abs2.(hilbert(f))[1:end÷2]

# make plot
using Plots,LaTeXStrings
begin
    # plt1 spacial domain, not too far away and the whole wake width
    plt1 = plot(size=(500,300),xlabel=L"y",ylabel=L"\zeta",labelfontsize=14)
    x=-8; y = range(-2+x/√8,2-x/√8,2^10)

    # plt2 spectral domain, further downstream and sampled along dg(t₊)=0
    plt2 = plot(size=(400,300),xlabel=L"k_y",ylabel=L"S^*_\zeta",labelfontsize=14)
    t = range(1/√2,8√50,2^14) # from the cusp to 8x the peak for z=-0.01
    xs = -40; ys = @. -xs*t/(1+2t^2)
    w =  @. √max(0,xs * (1 - 2t^2) / (1 + t^2)^(3/2))
    ky = @. t*hypot(1,t); ky = ky[1:end÷2]

    # Point-source over depths
    for n in 3:6
        z = -0.46416^n
        plot!(plt1,y,∂ₓW.(x,y,z),label="z=$(round(z, digits=3))",c=colormap("Blues",6)[n])
        f = ∂ₓW.(xs, ys, z) .- ∂ₓW(xs, 0, z)
        plot!(plt2,ky,spectrum(f),label="",c=colormap("Blues",6)[n])
    end

    # Elliptic line-source on z=-0.
    plot!(plt1,y,∂ₓWᵦ.(x,y,-0.),label="line-integrated z=0",c=:brown2,lw=2)
    fᵦ = ∂ₓWᵦ.(xs, ys, -0.) .- ∂ₓWᵦ(xs, 0, -0.)
    plot!(plt2,ky,spectrum(fᵦ),label="",c=:brown2,lw=2)

    # Finalize and save
    plot!(plt1, ylims=(-50,50), yticks=-50:25:50)
    plot!(plt2, yscale=:log10, ylims=(1e-3,1e3), yticks=10.0 .^ (-3:2:3))
    plot!(plt2, xscale=:log10, xticks=10.0 .^ (0:3))
    savefig(plt1,"wavecut.png")
    savefig(plt2,"spectrum.png")
end