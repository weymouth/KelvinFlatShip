# Define the integral of the Kelvin Green's function along a line with an elliptical strength distribution
include("../src/pointsource.jl")
include("../src/specialdiff.jl")
using TupleTools
using NeumannKelvin: kₓ,nearfield
k(t) = t*kₓ(t)
γj(t,b) = abs(t)<√eps(typeof(abs(t))) ? b/2 : SpecialFunctions.besselj1(b*k(t))/k(t)
γhx(t,b) = SpecialFunctions.besselhx(1,1,b*k(t))/2k(t) # can't use near t=0
"""
∫₂wavelike(x,y,z) = ∫ √(1-(y′/b)^2) W(x,y-y′,z) dy′

Integral of the wavelike Green's function along the line `y′∈[-b,b]` with an elliptical strength distribution. The wavelike integral is defined as

W(x,y,z) = 4H(-x)∫exp(z (1+t^2)) sin(g(x,y,t))) dt

where `t∈[-∞,∞], `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`. The integral in `y′` is evaluated analytically to give

∫ √(1-(y′/b)^2) W dy′ = 4π H(-x)∫ J₁(bk)/k exp(z (1+t^2)) sin(g(x,y,t)) dt

where `J₁` is the Bessel function of the first kind and `k(t)=t√(1+t^2)`. This is better regularized than the original integral for z≈0⁻, but still requires the use of a complex path of integration to avoid wasting 1000s of function evaluations on the decaying tail oscillations. Away from the stationary points and t=0 we use the identity

    J₁(z) = (H₁¹(z) exp(iz) + H₁¹(-z) exp(-iz))/2

where `H₁¹` is the scaled Hankel function of the first kind. This function is treated as slowly varying prefactors while the exponentials are absorbed into the complex steepest descent path.
"""
function ∫₂wavelike(x,y,z;b=1,ltol=-10,Δg=6)
    x,y,z = promote(x,y,z)
    (x≥0 || z≤ltol) && return zero(x)                            # no waves
    abs(y) > b-x/√8 && return π*wavelike(x,abs(y),z,γ=t->γj(t,b)) # fast treatment outside the wake

    # Find the stationary points and the finite ranges around them
    xv,yv,zv = value.((x,y,z))            # strip Duals
    R,atol = √(ltol/zv-1),10exp(ltol)     # angle limit and absolute tolerance
    rngs = mapreduce(vcat,(-b,b)) do y′   # merge three phase ranges
        S = TupleTools.sort(filter(s->-R<s<R,(S₀(xv,yv+y′)...,zero(xv))))
        rng = finite_ranges(S,t->g(xv,yv+y′,t),Δg,R;atol)
        rng2int(first.(rng))
    end |> sort |> merge

    # Real-line integrand and complex path phases and pre-factors
    f(t) = γj(t,b)*exp(z*(1+t^2))*sin(g(x,y,t))
    g₊(t)=g(x,y+b,t)-im*z*(1+t^2); dg₊(t)=dg(x,y+b,t)-2im*z*t; γ₊(t)=γhx(t,b)
    g₋(t)=g(x,y-b,t)-im*z*(1+t^2); dg₋(t)=dg(x,y-b,t)-2im*z*t; γ₋(t)=γhx(t,-b)

    # Sum over finite ranges and semi-infinite tails
    4π*sum(rngs) do (t₁,t₂)
        quadgk(f,t₁,t₂)[1] + sign(y+b)*(nsd(t₁,g₊,dg₊,γ₊)+nsd(t₁,g₋,dg₋,γ₋)) + sign(b-y)*(nsd(t₂,g₊,dg₊,γ₊)+nsd(t₂,g₋,dg₋,γ₋))
    end
end
rng2int(rngs::NTuple{N}) where N = [(rngs[2i-1],rngs[2i][1]) for i in 1:N÷2]
merge(intervals) = foldl(intervals[2:end], init=intervals[1:1]) do acc, (a,b)
    a > acc[end][2] ? push(acc, (a,b)) : (acc[end] = (acc[end][1], max(acc[end][2], b)); acc)
end

# Brute-force version for comparison
∫₂Wₜ(x,y,z,t) = γj(t,1)*exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t))
brute∫₂wavelike(x,y,z) = x ≥ 0 ? zero(x) : 4π*quadgk(t->∫₂Wₜ(x,y,z,t),-Inf,0,Inf)[1]

# Check the Bessel function integral identity is correct for an easy value of z
begin
    x,y,z = -1.,0.5,-1.
    @assert isapprox(brute∫₂wavelike(x,y,z),quadgk(y′->√(1-y′^2)*NeumannKelvin.wavelike(x,abs(y-y′),z),-1,1)[1],rtol=1e-7)
end

# Check the two ∫₂wavelike implementations give the same answer and compare timings
function check(y,x=-1.,z=-0.)
    wavelike = @btimed ∫₂wavelike($x,$y,$z) seconds=0.1
    brute = @timed brute∫₂wavelike(x,y,z)
    println("y = $y: wavelike = $(wavelike.value), brute = $(brute.value), wavelike time = $(wavelike.time) seconds, brute time = $(brute.time) seconds")
    (y=y, abserror = abs(wavelike.value-brute.value), relerror = abs(wavelike.value/brute.value-1), speedup = brute.time/wavelike.time)
end
flatship_table()=Table(check(y) for y in (0.,0.5,0.9,1.1,2.,4.))

# Add the near-field contribution using direct integration
∫₂kelvin(x,y,z,b=1) = ∫₂wavelike(x,y,z;b)+quadgk(y′->√(1-(y′/b)^2)*nearfield(x,y-y′,z),-b,b;atol=1e-4)[1]

# using Plots
# contour(-20:0.1:1,-10:0.1:10,(x,y)->∫₂kelvin(x,y,-0.),levels=-11:2:11,colormap=:phase,clims=(-12,12))
# contour(-20:0.1:1,-10:0.1:10,(x,y)->derivative(x->∫₂kelvin(x,y,-0.),x),levels=-11:2:11,clims=(-12,12))