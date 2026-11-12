# Which field of one step differs between TDGL.step! and TDGLGPU (CPU backend): the phases of the step one by one.
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Printf
BLAS.set_num_threads(1)
const G = TDGLGPU
backend = KernelAbstractions.CPU()
f = joinpath(@__DIR__, "work", "case.toml"); mkpath(dirname(f))
write(f, """
[material]
file = "mgb2_model_II.toml"
kappa = 2.0
[mesh]
nx = 24
ny = 12
nz = 6
dx = 0.5
[geometry]
GEOM_MODE = "WEDGE"
[scheme]
threads = false
$(get(ARGS, 1, ""))
[run]
J = [0.3]
H = 0.03
""")
c = load_config(f); st = Stepper(c.model, c.grid, c.masks; c.scheme...); HJ = TDGL.selffield(st, 0.3)
s = initial_state(c.model, c.grid, c.masks, c.H)
for _ in 1:5; step!(s, st; H=c.H, HJ=HJ); end
gs = GPUStepper(st, backend); g = device_state(backend, s)
d(a, b) = maximum(abs, Array(a) .- Array(b))
show(name, q...) = (KernelAbstractions.synchronize(backend); println(rpad(name, 22), join((@sprintf("%s %.2e", String(x), d(getfield(g, x), getfield(s, x))) for x in q), "  ")))
showb(name) = (KernelAbstractions.synchronize(backend); println(rpad(name, 22), join((@sprintf("F%d %.2e", µ, d(g.F[µ], s.F[µ])) for µ in 1:2), "  ")))
TDGL.update_bands!(s, st); G._update_bands!(g, gs); showb("bands")
TDGL.supercurrent!(s, st); G._supercurrent!(g, gs); show("supercurrent", :Jx, :Jy, :Jz)
TDGL.field_increment!(s, st); G._field_increment!(g, gs); show("field increment", :φx, :φy, :φz)
TDGL.implicit_curl!(s, st.imp); G._implicit_curl!(g, gs.sep, gs); show("implicit", :φx, :φy, :φz)
TDGL.links!(s, st); G._links!(g, gs); show("links", :Ux, :Uy, :Uz)
TDGL.curl_h!(s, st.grid); G._curl_h!(g, gs); show("curl", :hx, :hy, :hz)
TDGL.field_bc!(s, st.grid, c.H, HJ; bar=st.bar, open=st.open); G._field_bc!(g, gs, c.H, HJ); show("field bc", :hx, :hy, :hz)
TDGL.neumann!(s, st); G._neumann!(g, gs); showb("neumann")
