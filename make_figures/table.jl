# Accuracy and timing benchmarks for ∫₂wavelike vs brute-force quadrature
# Reproduces the table in "Linear Kelvin Wave predictions as z→0"
include("../src/specialdiff.jl")
include("../src/flatship.jl")
using BenchmarkTools, TypedTables

function flatship_check(y, x=-1., z=-0.)
    Wᵦ   = @btimed ∫₂wavelike($x,$y,$z,Δg=7) seconds=0.1
    W    = @btimed wavelike($x,$y-1,$z) seconds=0.1
    brute = @timed brute∫₂wavelike(x,y,z)
    println("y = $y: Wᵦ = $(Wᵦ.value), brute = $(brute.value), " *
            "Wᵦ time = $(Wᵦ.time) s, brute time = $(brute.time) s, " *
            "W time = $(W.time) s")
    (y        = y,
     abserror = abs(Wᵦ.value - brute.value),
     relerror = abs(Wᵦ.value / brute.value - 1),
     time     = Wᵦ.time,
     speedup  = brute.time / Wᵦ.time,
     slowdown = Wᵦ.time / W.time)
end

Table(flatship_check(y) for y in (0., 0.5, 0.9, 1.1, 1.35))