# Fix automatic differentiation of bessel functions with complex Dual inputs
using SpecialFunctions
using ForwardDiff: value, partials, Dual, derivative
function SpecialFunctions.besselj1(z::Complex{<:Dual{Tag}}) where {Tag}
    x, y = reim(z); px, py = partials(x), partials(y)
    w = complex(value(x), value(y))
    Ω = SpecialFunctions.besselj1(w)
    ∂Ω = SpecialFunctions.besselj0(w) - Ω/w  # dJ₁/dz = J₀(z) - J₁(z)/z
    u, v = reim(Ω); ∂u, ∂v = reim(∂Ω)
    complex(Dual{Tag}(u, ∂u*px - ∂v*py), Dual{Tag}(v, ∂v*px + ∂u*py))
end
function SpecialFunctions.besselhx(ν::Integer, k::Integer, z::Complex{<:Dual{Tag}}) where {Tag}
    x, y = reim(z); px, py = partials(x), partials(y)
    w = complex(value(x), value(y))
    Ω = SpecialFunctions.besselhx(ν, k, w)
    # d/dz besselhx(ν,k,z) = besselhx(ν-1,k,z) - (ν/z ± im)*besselhx(ν,k,z), +im for k=1, -im for k=2
    ∂Ω = SpecialFunctions.besselhx(ν-1, k, w) - (ν/w + im*(3-2k)) * Ω
    u, v = reim(Ω); ∂u, ∂v = reim(∂Ω)
    complex(Dual{Tag}(u, ∂u*px - ∂v*py), Dual{Tag}(v, ∂v*px + ∂u*py))
end
#test against finite difference approximation
delta(f,x;h=10√eps(typeof(abs(x)))) = (f(x+h)-f(x-h))/(2h)
z = 2. + im
@assert derivative(d->SpecialFunctions.besselj1(z+d),0.) ≈ delta(t->SpecialFunctions.besselj1(t), z)
@assert derivative(d->SpecialFunctions.besselhx(1,1,z+d),0.) ≈ delta(t->SpecialFunctions.besselhx(1,1,t), z)
@assert derivative(d->SpecialFunctions.besselhx(1,2,z+d),0.) ≈ delta(t->SpecialFunctions.besselhx(1,2,t), z)
