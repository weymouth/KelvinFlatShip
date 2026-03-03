# wave stuff
include("../src/pointsource.jl")
∂ₓW(x,y,z) = derivative(x->wavelike(x,y,z),x)
include("../src/flatship.jl")
∂ₓWᵦ(x,y,z) = derivative(x->∫₂wavelike(x,y,z)*2/π,x)

# spectrum stuff
using FFTW
function spectrum(y,f)
    dy,N = step(y),length(y)
    w = @. 0.5*(1 + cos(pi*(0:N-1)/(N-1)))
    S = abs2.(fft(f.*w)) * dy^2
    k = fftfreq(N, 1/dy) .* 2π
    k[2:end÷2],S[2:end÷2]
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
using Plots,LaTeXStrings
begin
    # Figure set up
    plt1 = plot(size=(500,300),xlabel=L"y",ylabel=L"\zeta")
    plt2 = plot(size=(400,300),xlabel=L"k_y",ylabel=L"S_\zeta")

    # Point-source over depths
    for n in 3:6
        z = -0.46416^n
        # plt[1] spacial domain, not too far away and the whole width
        x=-8; y = range(-2+x/√8,2-x/√8,2^10)
        plot!(plt1,y,∂ₓW.(x,y,z),label="z=$(round(z, digits=3))",c=colormap("Blues",6)[n])

        # plt[2] spectral domain, much farther and only through 1/4 of the width
        x=-40; y = range(0,-x/4√2,2^14)
        plot!(plt2,gbin(spectrum(y,∂ₓW.(x,y,z)))...,label="",c=colormap("Blues",6)[n])
    end

    # Elliptic line-source on z=-0.
    z=-0.; x=-8.; y = range(-2+x/√8,2-x/√8,2^10)
    plot!(plt1,y,∂ₓWᵦ.(x,y,z),label="line-integrated z=0",c=:brown2,lw=2)
    x=-40; y = range(0,-x/4√2,2^14)
    plot!(plt2,gbin(spectrum(y,∂ₓWᵦ.(x,y,z)))...,label="",c=:brown2,lw=2)

    # Finalize and save
    plot!(plt1, ylims=(-50,50), yticks=-50:25:50)
    plot!(plt2, yscale=:log10, xscale=:log10, xlims=(0.5,1e4), xticks=[1e0,1e2,1e4]) 
    plot!(plt2,[1,5e3],[1,(5e3)^(-2)], label="", ls=:dash, c=:grey)
    savefig(plt1,"wavecut.png")
    savefig(plt2,"spectrum.png")
end