# ============================================================================
#  step.jl — one full time step
# ============================================================================

"""
    step!(s, st; H, HJ, tilt=0) -> s

One step dt of every band and of the field: band update (with substeps), supercurrent, field
increment (implicit curl-curl if the stepper has it), links, curl, field BCs and injection,
Neumann ghosts, advance.  `H` is the applied field and `HJ` the self-field of the injected
current (`selffield(st, J)`).
"""
function step!(s::State{N}, st::Stepper{N}; H::Float64, HJ::Float64, tilt::Float64=0.0) where N
    st.open === nothing || open_update!(st.open, s, st.grid, st.model.κ^2, HJ)
    st.qpx === nothing || qp_potential!(st.qpx, s, st)
    sk = !isempty(st.ws.Qξ)                  # noise on a KWT band with RKC stages: the draws before the bands (SK-ROCK, noise.jl)
    sk && noise_draw!(st.noise, s, st)
    update_bands!(s, st)
    st.qpx === nothing || qp_turn!(st.qpx, s, st)
    st.noise === nothing || (sk ? noise_apply!(st.noise, s, st) : noise_kick!(st.noise, s, st))
    supercurrent!(s, st)
    st.heat === nothing || heat_step!(s, st)
    field_increment!(s, st)
    st.noise !== nothing && st.noise.field && field_kick!(st.noise, s, st)
    if st.imp !== nothing
        implicit_curl!(s, st.imp)
    elseif st.D !== nothing
        scale_increment!(s, st.D)
    end
    links!(s, st)
    curl_h!(s, st.grid)
    field_bc!(s, st.grid, H, HJ; tilt=tilt, bar=st.bar, open=st.open)
    neumann!(s, st)
    advance!(s, st.grid)
end

"""Self-field on the z faces for an injected current density J (Ampère, slab of thickness d)."""
selffield(st::Stepper, J::Real) = J * slab_thickness(st.grid) / (2 * st.model.κ^2)
