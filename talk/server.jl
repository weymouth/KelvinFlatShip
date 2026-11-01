# Wave fields for the live demo slide (js/demo.js).
#   julia --project=talk --threads=auto talk/server.jl           # serve on localhost:8081
#   julia --project=talk --threads=auto talk/server.jl --bake    # save fallback frames to assets/demo/
include(joinpath(@__DIR__, "..", "src", "flatship.jl"))
using ForwardDiff: derivative
import AcceleratedKernels as AK
using HTTP

nearline(x,y,b) = quadgk(y′->√(1-(y′/b)^2)*nearfield(x,y-y′,-0.),-b,b;atol=1e-4)[1]
ζb(x,y,b) = derivative(x->∫₂wavelike(x,y,-0.;b)+nearline(x,y,b),x)*2/(π*b)
ζp(x,y,z) = derivative(x->wavelike(x,y,z)+nearfield(x,y,z),x)
bruteζb(x,y,b) = (x ≥ 0 ? 0. : 4π*quadgk(t->γj(t,b)*cos(g(x,y,t))*hypot(1,t),-Inf,0,Inf,atol=10exp(-10),maxevals=7*10^6)[1]    # same atol as ∫₂wavelike; quadgk's estimate is unreliable at z=0, so cap it
                  + derivative(x->nearline(x,y,b),x))*2/(π*b)
bruteζp(x,y,z) = (x ≥ 0 ? 0. : quadgk(t->4exp(z*(1+t^2))*cos(g(x,y,t))*hypot(1,t),-Inf,0,Inf,atol=10exp(-10),maxevals=7*10^6)[1]
                  + derivative(x->nearfield(x,y,z),x))

# Window in Kelvin lengths for Froude number F: bow at x=0, columns run from the bow
# (x = 0.1X) downstream, rows from y = +X/4 to -X/4. The stern is nL columns behind.
function window(F; nx=160, ny=80)
    L = 1/F^2; X = max(4L, 12); h = 1.1X/nx; nL = round(Int, L/h)
    x(i) = 0.1X-(i-0.5)h; y(j) = (ny/2-j+0.5)h
    (; nx, ny, nL, x, y)
end

# Field of a bow source minus a stern source, from one kernel grid shifted by nL columns
function frame(f, F)
    (; nx, ny, nL, x, y) = window(F)
    K = Array{Float64}(undef, nx+nL, ny÷2)
    AK.foraxes(K, 1) do i; for j in axes(K, 2)
        K[i,j] = f(x(i-nL), y(j))
    end; end
    H = K[nL+1:end,:]-K[1:nx,:]
    vec(permutedims(hcat(H, reverse(H, dims=2)))), nx, ny
end

# Pixels k0+1:k0+n of the same frame (column-major from the bow) by direct quadrature
function brute(f, F, k0, n)
    (; nx, ny, nL, x, y) = window(F)
    k = k0+1:min(k0+n, nx*ny); z = zeros(length(k))
    for m in eachindex(k)
        i, j = divrem(k[m]-1, ny) .+ 1; j = min(j, ny+1-j)
        z[m] = f(x(i), y(j))-f(x(i-nL), y(j))
    end; z
end

json(z; kw...) = "{" * join(["\"$k\":$v" for (k,v) in kw] ∪ ["\"z\":[" * join(round.(clamp.(z,-20,20), digits=2), ",") * "]"], ",") * "}"

function handle(req)
    u = HTTP.URI(req.target); q = HTTP.queryparams(u)
    F, β = parse(Float64, q["F"]), parse(Float64, q["beta"]); b = β/(2F^2)
    point = get(q, "model", "flat") == "point"; zs = min(parse(Float64, get(q, "z", "0")), -1e-3)
    body = if u.path == "/field"
        f = point ? (x,y)->ζp(x, y, zs) : (x,y)->ζb(x, y, b)
        time = @elapsed (z, nx, ny) = frame(f, F)
        kbow = count(i->window(F).x(i) ≥ 0, 1:nx)*ny    # pixels ahead of the bow are still water
        json(z; nx, ny, kbow, time)
    else
        f = point ? (x,y)->bruteζp(x, y, zs) : (x,y)->bruteζb(x, y, b)
        json(brute(f, F, parse(Int, q["k0"]), parse(Int, q["n"])))
    end
    HTTP.Response(200, ["Access-Control-Allow-Origin"=>"*", "Content-Type"=>"application/json"], body)
end

frame((x,y)->ζb(x,y,0.5), 0.6); brute((x,y)->bruteζb(x,y,0.5), 0.6, 0, 1)    # compile before the first request
if "--bake" in ARGS
    mkpath(joinpath(@__DIR__, "assets", "demo"))
    for F in 0.25:0.05:0.8, β in 0.1:0.05:0.6
        local z, nx, ny
        time = @elapsed (z, nx, ny) = frame((x,y)->ζb(x,y,β/(2F^2)), F)
        f2(v) = (n = round(Int, 100v); "$(n ÷ 100).$(lpad(n % 100, 2, '0'))")    # as JS toFixed(2)
        write(joinpath(@__DIR__, "assets", "demo", "F_$(f2(F))_beta_$(f2(β)).json"), json(z; nx, ny, time))
    end
else
    println("Serving wave fields on http://localhost:8081 with $(Threads.nthreads()) threads")
    HTTP.serve(handle, "127.0.0.1", 8081)
end
