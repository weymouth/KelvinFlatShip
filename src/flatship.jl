# Define the integral of the Kelvin Green's function along a line with an elliptical strength distribution
using TupleTools
using NeumannKelvin: kₓ,nearfield
k(t) = t*kₓ(t)
γ(t,b) = abs(t)<√eps(typeof(abs(t))) ? b/2 : SpecialFunctions.besselj1(b*k(t))/k(t)
γ1(t,b) = SpecialFunctions.besselhx(1,1,b*k(t))/2k(t) # can't use near t=0
γ2(t,b) = SpecialFunctions.besselhx(1,2,b*k(t))/2k(t) # can't use near t=0
"""
∫₂wavelike(x,y,z) = ∫ √(1-(y′/b)^2) W(x,y-y′,z) dy′

Integral of the wavelike Green's function along the line `y′∈[-b,b]` with an elliptical strength distribution. The wavelike integral is defined as

W(x,y,z) = 4H(-x)∫exp(z (1+t^2)) sin(g(x,y,t))) dt

where `t∈[-∞,∞], `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`. The integral in `y′` is evaluated analytically to give

∫ √(1-(y′/b)^2) W dy′ = 4π H(-x)∫ J₁(bk)/k exp(z (1+t^2)) sin(g(x,y,t)) dt

where `J₁` is the Bessel function of the first kind and `k(t)=t√(1+t^2)`. This is better regularized than the original integral for z≈0⁻, but still requires the use of a complex path of integration to avoid wasting 1000s of function evaluations on the decaying tail oscillations. Away from the stationary points and t=0 we use the identity

    J₁(z) = (H₁¹(z) exp(iz) + H₁²(z) exp(-iz))/2

where `H₁¹²` are the Hankel functions of the first and second kind, respectively. These functions are treated as slowly varying prefactors while the exponentials are absorbed into the complex steepest descent path.
"""
function ∫₂wavelike(x,y,z,b=1,ltol=-10,Δg=12)
    x,y,z = promote(x,abs(y),z)
    (x≥0 || z≤ltol) && return zero(x) # no waves
    abs(y) > b-x/√8 && return π*wavelike(x,y,z,γ=t->γ(t,b)) # fast treatment outside the wake

    # Find the stationary points and the finite ranges around them
    xv,yv,zv = value.((x,y,z))         # strip Duals
    S = TupleTools.sort((zero(xv),S₀(xv,yv)...,S₀(xv,yv-b)...,S₀(xv,yv+b)...))
    R,atol = √(ltol/zv-1),10exp(ltol)  # angle limit and absolute tolerance
    rngs = finite_ranges(filter(s->-R<s<R,S),t->g(xv,yv,t),Δg,R;atol)

    # Real-line integrand and complex path phases and pre-factors
    f(t) = γ(t,b)*exp(z*(1+t^2))*sin(g(x,y,t))
    g₊(t)=g(x,y+b,t)-im*z*(1+t^2); dg₊(t)=dg(x,y+b,t)-2im*z*t; γ₊(t)=γ1(t,b)
    g₋(t)=g(x,y-b,t)-im*z*(1+t^2); dg₋(t)=dg(x,y-b,t)-2im*z*t; γ₋(t)=γ2(t,b)

    # Sum over finite ranges and semi-infinite tails
    val = zero(f(zero(x)))
    for i in 1:2:length(rngs)
        (t₁,∞₁),(t₂,∞₂) = rngs[i],rngs[i+1]
        ∞₁ && (val -= nsd(t₁,g₊,dg₊,γ₊) + nsd(t₁,g₋,dg₋,γ₋))
        val += quadgk(f,t₁,t₂;atol)[1]
        ∞₂ && (val += nsd(t₂,g₊,dg₊,γ₊) + nsd(t₂,g₋,dg₋,γ₋))
    end
    return 4π*val
end

# Real-line integrand (use in quadgk and plotting)
function ∫₂W(x,y,z,t)
    kx = hypot(1,t); ky = t*kx; kz = 1+t^2
    γ(t,1)*exp(z*kz)*sin(x*kx+y*ky)
end

# Brute-force version for comparison
brute∫₂wavelike(x,y,z) = x ≥ 0 ? zero(x) : 4π*quadgk(t->∫₂W(x,y,z,t),-Inf,0,Inf)[1]

# Check the Bessel function integral identity is correct for an easy value of z
x,y,z = -1.,0.5,-1.
quadgk_count(t->4π*∫₂W(x,y,z,t),-Inf,0,Inf)
quadgk_count(y′->√(1-y′^2)*wavelike(x,abs(y-y′),z),-1,1)

# Check the two ∫₂wavelike implementations give the same answer and compare timings
function check(y,x=-1.,z=-0.)
    wavelike = @timed ∫₂wavelike(x,y,z)
    brute = @timed brute∫₂wavelike(x,y,z)
    println("y = $y: wavelike = $(wavelike.value), brute = $(brute.value), wavelike time = $(wavelike.time) seconds, brute time = $(brute.time) seconds")
    (y=y, kv = wavelike.value, bv = brute.value, kt = wavelike.time, bt = brute.time)
end
Table(check,(0.,0.5,1.,2.,4.))

# Add the near-field contribution using direct integration 
∫₂kelvin(x,y,z,b=1) = ∫₂wavelike(x,y,z;b)+quadgk(y′->√(1-(y′/b)^2)*nearfield(x,y-y′,z),-b,b;atol=1e-4)[1]

using Plots
contour(-20:0.1:1,-10:0.1:10,(x,y)->∫₂kelvin(x,y,-0.),levels=-11:2:11,colormap=:phase,clims=(-12,12))
contour(-20:0.1:1,-10:0.1:10,(x,y)->derivative(x->∫₂kelvin(x,y,-0.),x),levels=-11:2:11,clims=(-12,12))