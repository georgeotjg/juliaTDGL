# ---------------------------------------------------------------------------
#  Geometry.  What is tested is not "that it draws something": it is the
#  three properties on which the symmetry CONTROLS depend.
# ---------------------------------------------------------------------------
@testset "geometry: masks, mirror and margin" begin
    cfg = cfg_fixture(); p = derive_params(cfg); g = build_geometry(cfg, p)
    (; x1, x2, y1, y2, z1, z2, Nx, Ny, Nz) = p

    # --- masks are STAGGERED by link type --------------------------------------
    # A single mask for the three directions excludes the high-index transverse
    # rows (j=y2+1, k=z2+1) and includes the low-index ones, biasing the Maxwell
    # source and the voltmeter towards one edge: a spurious +J/-J asymmetry of
    # up to 14 %.
    @test size(g.taux) == (Nx+1, Ny+1, Nz+1)
    @test all(g.taux[x1:x2, y1:y2+1, z1:z2+1] .== 1)
    @test sum(g.taux) == (x2-x1+1)*(y2-y1+2)*(z2-z1+2)
    @test sum(g.tauy) == (x2-x1+2)*(y2-y1+1)*(z2-z1+2)
    @test sum(g.tauz) == (x2-x1+2)*(y2-y1+2)*(z2-z1+1)
    @test sum(g.taupsi) == (x2-x1+2)*(y2-y1+2)*(z2-z1+2)

    # --- the WEDGE notch exists, touches face y1 and not the x faces ----------
    @test minimum(g.fmask) == 0.0
    @test any(g.fmask[:, y1, :] .== 0)                # notch on face y1
    @test all(g.fmask[:, y2, :] .== 1)                # face y2 FLAT: that is the diode
    @test all(g.fmask[x1, :, :] .== 1)                # clean injection face
    @test all(g.fmask[x2+1, :, :] .== 1)

    # --- the _REF mirror is EXACT on the interior nodes ------------------------
    # This makes the control a control: if _REF is not the mirror, an eta that
    # "reverses" proves nothing.  The reflection is j -> (y1+y2+1)-j, not
    # (y1+y2)-j, which reflects and also shifts by one cell.
    cref = cfg_fixture(GEOM_MODE="WEDGE_REF"); pr = derive_params(cref)
    gr = build_geometry(cref, pr)
    dif = 0
    for k in z1:z2, j in y1+1:y2, i in x1+1:x2
        gr.fmask[i,j,k] == g.fmask[i, (y1+y2+1)-j, k] || (dif += 1)
    end
    @test dif == 0
    @test check_my_symmetry(g, p) > 0        # WEDGE is NOT M_y-symmetric...
    @test any(gr.fmask[:, y2, :] .== 0) || any(gr.fmask[:, min(y2+1, size(gr.fmask,2)), :] .== 0)
    cs = cfg_fixture(GEOM_MODE="SIM"); ps = derive_params(cs)
    @test check_my_symmetry(build_geometry(cs, ps), ps) == 0   # ...SIM is
    cfun = Config(:legacy; GEOM_MODE="FUNNEL", Nx_sc=60, Ny_sc=24, Nz_sc=6, dx=0.5,
                  prism_phi=0.0, verbose=false, selftest=false, out_dir="")
    @test build_geometry(cfun, derive_params(cfun)) isa Geometry

    # --- Octave-style rounding ---------------------------------------------------
    # round(x.5) goes away from zero in Octave and to even in Julia.  One cell
    # of difference changed the voltage by 9.4 %.
    @test TDGL.oround(0.5) == 1 && TDGL.oround(1.5) == 2 && TDGL.oround(2.5) == 3
    @test TDGL.oround(-0.5) == -1

    # --- the whole catalogue builds ----------------------------------------------
    for modo in ("SIM","ESFERA","ARRAY","WEDGE","WEDGE_BOTH","ESF_CUNA","FGRAD",
                 "SAWROW","SPHROW","SAWSPH","STAGGER","FUNNEL","FUNNEL_IN",
                 "TRI_Y1","TRI_Y1_INV","RECT_Y1","HEX","HEX_STAG")
        c = Config(:legacy; GEOM_MODE=modo, Nx_sc=60, Ny_sc=24, Nz_sc=6,
                   dx=0.5, dy=0.5, dz=0.5, wedge_depth2=0.20,
                   verbose=false, out_dir="")
        pp = derive_params(c)
        @test build_geometry(c, pp) isa Geometry
    end
    @test_throws ErrorException build_geometry(Config(:legacy; GEOM_MODE="NOEXISTE",
        Nx_sc=20, Ny_sc=10, Nz_sc=4, verbose=false), derive_params(Config(:legacy;
        GEOM_MODE="SIM", Nx_sc=20, Ny_sc=10, Nz_sc=4, verbose=false)))

    # --- the n-gon: inside the apothem yes, outside the circumradius no -------
    @test in_ngon(0, 0, 0, 0, 5, 3, 0.0)
    @test !in_ngon(10, 0, 0, 0, 5, 3, 0.0)

    # cos(π/3)*6 lands 1 ulp high; without +1e-9 the base row is lost (10 vs 11).
    @test in_ngon(0.0, -3.0, 0.0, 0.0, 6.0, 3, pi/2)
    nbase = count(i -> in_ngon(Float64(i), -3.0, 0.0, 0.0, 6.0, 3, pi/2), -6:6)
    @test nbase >= 11

    # Symmetry theorems: SIM has My=Mx=Rz=0; a diode must break My and Rz.
    csim = Config(:legacy; GEOM_MODE="SIM", Nx_sc=40, Ny_sc=24, Nz_sc=6,
                  dx=0.5, dy=0.5, dz=0.5, verbose=false, selftest=false, out_dir="")
    psim = derive_params(csim); gsim = build_geometry(csim, psim)
    @test check_my_symmetry(gsim, psim) == 0
    @test check_mx_symmetry(gsim, psim) == 0
    @test check_rz_symmetry(gsim, psim) == 0
    @test check_my_symmetry(g, p) > 0          # WEDGE of the fixture
    @test check_rz_symmetry(g, p) > 0
    cfun = Config(:legacy; GEOM_MODE="FUNNEL", Nx_sc=40, Ny_sc=24, Nz_sc=6,
                  dx=0.5, dy=0.5, dz=0.5, verbose=false, selftest=false, out_dir="")
    pfun = derive_params(cfun); gfun = build_geometry(cfun, pfun)
    @test check_my_symmetry(gfun, pfun) > 0
    @test check_rz_symmetry(gfun, pfun) > 0
end
