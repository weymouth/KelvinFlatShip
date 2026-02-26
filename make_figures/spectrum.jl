# wave stuff
include("../src/pointsource.jl")
∂ₓW(x,y,z) = derivative(x->wavelike(x,y,z),x)
include("../src/flatship.jl")
∂ₓWᵦ(x,y,z) = derivative(x->∫₂wavelike(x,y,z),x)

# spectrum stuff
using FFTW
function spectrum(y,f)
    dy,N = step(y),length(y)
    w = @. 0.5*(1 + cos(pi*(0:N-1)/(N-1)))
    S² = abs2.(fft(f.*w)) * dy^2
    k = fftfreq(N, 1/dy) .* 2π
    k[2:end÷2],S²[2:end÷2]
end
function gbin((k,S); r=1.07)
    nbins = ceil(Int,log(1-k[end]/k[1]*(1-r))/log(r))-1
    edges = k[1] .* cumsum(r .^ (0:nbins)) .- k[1]/2
    kbin = map(i->gmean(edges[i:i+1]),1:nbins)
    Sbin = map(i->gmean(S[edges[i] .< k .≤ edges[i+1]]),1:nbins)
    return kbin, Sbin
end
gmean(a) = log.(a) |> mean |> exp

# make plot
using Plots
begin
    # Figure set up
    plt = plot(layout=(2,1), size=(800,800), guidefontpadding=2)
    plot!(plt[1],xlabel="y",ylabel="∂ₓW")
    plot!(plt[2],xlabel="k",ylabel="|S|²", yscale=:log10, xscale=:log10)

    # Point-source over depths
    for n in 3:6
        z = -0.46416^n
        # plt[1] spacial domain, not too far away and the whole width
        x=-8; y = range(-2+x/√8,2-x/√8,2^10)
        plot!(plt[1],y,∂ₓW.(x,y,z),label="z=$(round(z, digits=3))",c=colormap("Blues",6)[n])

        # plt[2] spectral domain, much farther and only through 1/4 of the width
        x=-40; y = range(0,-x/4√2,2^14)
        plot!(plt[2],gbin(spectrum(y,∂ₓW.(x,y,z)))...,label="",c=colormap("Blues",6)[n])
    end

    # Elliptic line-source on z=-0.
    z=-0.; x=-8.; y = range(-2+x/√8,2-x/√8,2^10)
    plot!(plt[1],y,∂ₓWᵦ.(x,y,z),label="line-integrated z=0",c=:forestgreen)
    x=-40; y = range(0,-x/4√2,2^14)
    plot!(plt[2],gbin(spectrum(y,∂ₓWᵦ.(x,y,z)))...,label="",c=:forestgreen)
end;plt
savefig(plt,"point_spectrum.png")