# Extra renders for the talk, in the style of make_figures/wave_field.jl:
#   julia --project=talk --threads=auto talk/scripts/renders.jl
include(joinpath(@__DIR__, "..", "..", "src", "flatship.jl"))
using ForwardDiff: derivative
import AcceleratedKernels as AK
using Plots, LaTeXStrings

ζp(x,y,z) = derivative(x->wavelike(x,y,z)+nearfield(x,y,z),x)
function field(f,x,y)
    z = Array{Float64}(undef,length(x),length(y))
    AK.foraxes(z,1) do i; for j in axes(z,2)
        z[i,j] = f(x[i],y[j])
    end; end; z
end
img(name) = joinpath(@__DIR__, "..", "assets", "img", name)

# Slide 11: one submerged point source, framed like the graphical abstract. The colour range
# is 3× the field's 99th percentile, so a single source stays in the pale shades.
begin
    h = 0.025; x = -20:h:1; y = -10:h:10
    z = field((x,y)->ζp(x,y,-0.2), x, y)
    ζm = 3sort(abs.(vec(z)))[round(Int, 0.99length(z))]
    contourf(x, y, clamp.(z,-ζm,ζm)', levels=18, clims=(-ζm,ζm), colormap=:balance, line=0,
        xlims=(-20,1), ylims=(-10,10), axis=false, ticks=false, colorbar=false, framestyle=:none,
        size=(709,591), dpi=300, margin=0Plots.mm)
    scatter!([0], [0], c=:black, ms=4, msw=0, label="")
    savefig(img("point_wake.png"))
end

# Slide 12: make_figures/spectrum.jl's wave cut with the point sources only
begin
    ∂ₓW(x,y,z) = derivative(x->wavelike(x,y,z),x)
    x = -8; y = range(-2+x/√8, 2-x/√8, 2^10)
    plt = plot(size=(500,300), xlabel=L"y", ylabel=L"\zeta", labelfontsize=14, ylims=(-50,50), yticks=-50:25:50)
    for n in 3:6
        z = -0.46416^n
        plot!(plt, y, ∂ₓW.(x,y,z), label="z=$(round(z, digits=3))", c=colormap("Blues",6)[n])
    end
    savefig(plt, img("wavecut_points.png"))
end

# Results slides: slices of make_figures/wavedrag.jl's C_W(L,b), in the deck's colours.
# C_W is drag scaled by beam², so drag scaled by planform area 2bL is C_W 2b/L. Thin-ship
# theory keeps C_W fixed, so this grows linearly with b: drawn (cool, dashed) as the tangent
# to the flat-ship curve at the narrowest ship, since the flat-ship C_W diverges as b → 0.
Cw(L,b) = quadgk(t->(π*SpecialFunctions.besselj1(b*k(t))/b/k(t))^2*kₓ(t)*(1-cos(L*hypot(1,t))),0,Inf;atol=1e-6)[1]
slide(; kw...) = plot(; size=(900,450), legend=:right, background_color=:transparent, foreground_color=:gray30,
    guidefontsize=16, tickfontsize=13, legendfontsize=14, grid=false, framestyle=:axes, margin=4Plots.mm, bottom_margin=8Plots.mm, kw...)
warm, cool = colorant"#b3650a", colorant"#00739a"
begin
    L = 3π; r = range(0.02, 1, 100); D(r) = Cw(L,r*L/2)*r   # aspect ratio r = 2b/L
    d = D.(r); s = (D(0.025)-d[1])/0.005
    plt = slide(legend=:topleft, xlabel="width / length", ylabel="wave drag / ρU²(area)", ylims=(0, Inf))
    plot!(plt, r, d[1] .+ s*(r .- r[1]), c=cool, ls=:dash, lw=4, label="textbook (thin-ship) theory")
    plot!(plt, r, d, c=warm, lw=5, label="with width")
    savefig(plt, img("drag_vs_width.png"))
end
begin
    F = range(0.2, 1, 300); b = 1
    plt = slide(xlabel="speed (Froude number)", ylabel="wave drag", ylims=(0, Inf), legend=false)
    plot!(plt, F, Cw.(1 ./ F.^2, b), c=warm, lw=5)
    savefig(plt, img("drag_vs_length.png"))
end
