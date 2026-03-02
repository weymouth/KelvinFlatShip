# Define the integral of the Kelvin Green's function along a line with an elliptical strength distribution
using NeumannKelvin,QuadGK
using NeumannKelvin: finite_ranges, g, dg
using NeumannKelvin: nsp as nsd
using NeumannKelvin: stationary_points as S₀
using ForwardDiff: value
"""
wavelike(x,y,z,kwargs...) = 4H(-x) ∫ exp(z (1+t^2)) sin(g(x,y,t))) dt

Wavelike part of the Kelvin Green's function, where `t∈[-∞,∞], the phase `g(x,y,t)=(x+yt)√(1+t²)`, and `H` is the Heaviside function, forcing `W=0` when `x>0`.

The function is integrated using a modified steepest descent method. The stationary points are found and the smooth integral over finite ranges nearby is integrated using Gauss-Konrad quadrature, while the highly oscillatory tails are integrated using numerical steepest descent.

# Optional kwargs

- `γ=one`: Integrand weight-function which must be slowly varying compared to the exponential phase.
- `Δg=6`: Phase range for real-line integration around stationary points.
- `ltol=-10`: Log tolerance for finding stationary points and finite ranges around them.

"""
function wavelike(x,y,z;γ=one,Δg=6,ltol=-10,xlag=NeumannKelvin.xlag,wlag=NeumannKelvin.wlag)
    x,y,z = promote(x,abs(y),z)
    (x≥0 || z≤ltol) && return zero(x)         # no waves

    # Find the stationary points and the finite ranges around them
    xv,yv,zv = value.((x,y,z))                # strip Duals
    R,atol = √(ltol/zv-1),10exp(ltol)         # angle limit and absolute tolerance
    S = filter(s->-R<s<R,S₀(xv,yv))           # stationary points
    rngs = finite_ranges(S,t->g(xv,yv,t),Δg,R;atol)

    # Define the real-line integrand
    f(t) = γ(t)*exp(z*(1+t^2))*sin(g(x,y,t))  # integrand
    length(S)==0 && return 4quadgk(f,-R,R;atol)[1] # no stationary points, just integrate

    # Define the complex phase and its derivative
    G(t) = g(x,y,t)-im*z*(1+t^2); dG(t) = dg(x,y,t)-2im*z*t

    # Sum over finite ranges and semi-infinite tails
    val = zero(f(zero(x)))
    for i in 1:2:length(rngs)
        (t₁,∞₁),(t₂,∞₂) = rngs[i],rngs[i+1]
        ∞₁ && (val -= nsd(t₁,G,dG,γ;atol=2atol,xlag,wlag))
        val += quadgk(f,t₁,t₂;atol)[1]
        ∞₂ && (val += nsd(t₂,G,dG,γ;atol=2atol,xlag,wlag))
    end
    return 4val
end

# Brute-force version for comparison
brutewavelike(x,y,z;atol=0) = x ≥ 0 ? zero(x) : quadgk(t->4exp(z*(1+t^2))*sin(g(x,y,t)),-Inf,Inf;atol)[1]

# Check the two wavelike implementations give the same answer and compare timings
using BenchmarkTools
function check(y,x=-1.,z=-0.001)
    kelvin = @btimed wavelike($x,$y,$z) seconds=0.1
    brute = @btimed brutewavelike($x,$y-1,$z) seconds=0.1
    println("y = $y: kelvin = $(kelvin.value), brute = $(brute.value), kelvin time = $(kelvin.time) seconds, brute time = $(brute.time) seconds")
    (y=y, relerror = abs(kelvin.value/brute.value-1), speedup = brute.time/kelvin.time)
end
make_table() = Table(check(y) for y in (0.,0.1/sqrt(8.),0.125,1/sqrt(8.),1.25))

# using Plots
# contour(-20:0.1:1,-10:0.1:10,(x,y)->wavelike(x,y,-0.1),levels=-11:2:11,colormap=:phase,clims=(-12,12))
# contour(-20:0.1:1,-10:0.1:10,(x,y)->derivative(x->wavelike(x,y,-0.1),x),levels=-11:2:11,clims=(-12,12))