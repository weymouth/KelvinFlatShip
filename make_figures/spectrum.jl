using Plots
include("../src/pointsource.jl")
∂ₓW(x,y,z) = derivative(x->wavelike(x,y,z),x)
function bad_ζ(z,R=range(0.1,10,1000))
    θ = √(-2z)/2      # worst wavenumber ray
    R,∂ₓW.(-R.*cos(θ), R.*sin(θ),z)
end
plt = plot(xlabel="R",ylabel="∂ₓW",xflip=true);
for n in 3:8
    z = -0.5^n
    plot!(bad_ζ(z)...,label="z=1/2^$n")
end; plt

using NFFT
function spectrum_nfft(z; θ=range(0,π/9,2^10), R=range(0.1,100,2^9), Nk=(2^10,2^10))
    x = vec(-R .* cos.(θ')); y = vec(R .* sin.(θ'))
    R₁,y₁ = last(R),maximum(y)
    pnts = vcat(x' ./R₁ .+0.5,y' ./y₁ .-0.5)
    w = vec(R .* ones(length(θ))') .* step(R) .* step(θ)
    F = nfft_adjoint(pnts, Nk, complex.(∂ₓW.(x,y,z) .* w))
    kx = fftfreq(Nk[1], Nk[1]/R₁) .* 2π
    ky = fftfreq(Nk[2], Nk[2]/y₁) .* 2π
    kx[1:Nk[1]÷2] .^2, vec(sum(abs2.(F),dims=2))[1:Nk[1]÷2].*[ky[2]-ky[1]]
end
plt = plot(yscale=:log10,xlims=(0.1,100),xscale=:log10);
for n in 3:7
    z = -0.5^n
    k,S = spectrum_nfft(z)
    plot!(k[2:end],S[2:end],label="z=-1/2^$n")
end; plt