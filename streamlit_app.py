"""FElupe Cantilever Beam Example with Streamlit and stpyvista.

Interactive visualization of a cantilever beam under gravity load.
"""

import felupe as fem
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

st.set_page_config(page_title="FElupe Beam Example", layout="wide")

st.title("🏗️ Cantilever Beam Under Gravity")
st.markdown("""
This app demonstrates a cantilever beam subjected to gravity load using FElupe.
The beam is fixed at one end and deforms under its own weight.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Parameters")

# Mesh parameters
st.sidebar.subheader("Mesh Settings")
mesh_resolution = st.sidebar.select_slider(
    "Mesh Resolution",
    options=["Coarse", "Medium", "Fine"],
    value="Medium",
    help="Higher resolution provides more accurate results but takes longer to compute",
)

resolution_map = {
    "Coarse": (51, 4, 4),
    "Medium": (101, 6, 6),
    "Fine": (151, 8, 8),
}
n_elements = resolution_map[mesh_resolution]

# Material parameters
st.sidebar.subheader("Material Properties")
material_type = st.sidebar.selectbox(
    "Material",
    options=["Steel", "Aluminum", "Concrete", "Custom"],
    index=0,
)

material_properties = {
    "Steel": {"E": 206000, "nu": 0.30, "density": 7850e-12},
    "Aluminum": {"E": 70000, "nu": 0.33, "density": 2700e-12},
    "Concrete": {"E": 30000, "nu": 0.20, "density": 2400e-12},
}

if material_type == "Custom":
    E = st.sidebar.number_input("Young's Modulus E (MPa)", value=206000, min_value=1)
    nu = st.sidebar.number_input(
        "Poisson's Ratio v",
        value=0.3,
        min_value=0.0,
        max_value=0.49,
        step=0.01,
    )
    density = (
        st.sidebar.number_input("Density (kg/m³)", value=7850, min_value=1) * 1e-12
    )
else:
    props = material_properties[material_type]
    E = props["E"]
    nu = props["nu"]
    density = props["density"]
    st.sidebar.info(f"E = {E} MPa, v = {nu:.2f}, p = {density*1e12:.0f} kg/m³")

# Visualization parameters
st.sidebar.subheader("Visualization")
scale_factor = st.sidebar.slider(
    "Displacement Scale Factor",
    min_value=1,
    max_value=1000,
    value=300,
    step=50,
    help="Scale factor for visualizing displacements",
)

show_edges = st.sidebar.checkbox("Show mesh edges", value=True)
show_undeformed = st.sidebar.checkbox("Show undeformed shape", value=False)

# Solve button
if st.sidebar.button("🚀 Solve", type="primary", use_container_width=True):
    with st.spinner("Creating mesh and solving..."):
        # Create mesh and region
        cube = fem.Cube(a=(0, 0, 0), b=(2000, 100, 100), n=n_elements)
        region = fem.RegionHexahedron(cube, uniform=True)

        # Create displacement field
        displacement = fem.Field(region, dim=3)
        field = fem.FieldContainer([displacement])

        # Apply fixed boundary condition on left end
        boundaries = fem.BoundaryDict(fixed=fem.dof.Boundary(displacement, fx=0))

        # Define material
        umat = fem.LinearElastic(E=E, nu=nu)
        solid = fem.SolidBody(umat=umat, field=field)

        # Define gravity load
        gravity = [0, 0, 9810]  # mm/s²
        force = fem.SolidBodyForce(field, values=gravity, scale=density)

        # Solve
        step = fem.Step(items=[solid, force], boundaries=boundaries)
        job = fem.Job(steps=[step]).evaluate()

        # Store results in session state
        st.session_state.field = field
        st.session_state.region = region
        st.session_state.displacement = displacement
        st.session_state.solved = True

# Display results
if "solved" in st.session_state and st.session_state.solved:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Visualization")

        # Create PyVista mesh for visualization
        field = st.session_state.field
        region = st.session_state.region
        displacement = st.session_state.displacement

        # Get the deformed mesh
        points = region.mesh.points + scale_factor * displacement.values
        cells = region.mesh.cells

        # Create PyVista unstructured grid
        # FElupe cube mesh uses hexahedron elements
        cells_pv = []
        for cell in cells:
            cells_pv.extend([8, *list(cell)])
        mesh = pv.UnstructuredGrid(
            cells_pv,
            [pv.CellType.HEXAHEDRON] * len(cells),
            points,
        )

        # Add displacement magnitude
        disp_magnitude = (displacement.values**2).sum(axis=1) ** 0.5
        mesh["Displacement Magnitude"] = disp_magnitude

        # Create plotter
        plotter = pv.Plotter(window_size=[800, 600])
        plotter.add_mesh(
            mesh,
            scalars="Displacement Magnitude",
            cmap="coolwarm",
            show_edges=show_edges,
            edge_color="gray",
            lighting=True,
        )

        # Add undeformed shape if requested
        if show_undeformed:
            mesh_undeformed = pv.UnstructuredGrid(
                cells_pv,
                [pv.CellType.HEXAHEDRON] * len(cells),
                region.mesh.points,
            )
            plotter.add_mesh(
                mesh_undeformed,
                color="gray",
                opacity=0.3,
                show_edges=True,
                edge_color="darkgray",
            )

        # Add axes and set view
        plotter.show_axes()
        plotter.view_isometric()
        plotter.add_text(f"Scale Factor: {scale_factor}x", position="upper_left")

        # Display with stpyvista
        stpyvista(plotter, key="beam_viz")

    with col2:
        st.subheader("📈 Results Summary")

        # Calculate statistics
        max_disp = disp_magnitude.max()
        max_disp_z = displacement.values[:, 2].max()
        min_disp_z = displacement.values[:, 2].min()

        st.metric("Maximum Displacement", f"{max_disp:.3f} mm")
        st.metric("Max Vertical Displacement", f"{max_disp_z:.3f} mm")
        st.metric("Min Vertical Displacement", f"{min_disp_z:.3f} mm")

        st.divider()

        st.subheader("📐 Model Information")
        st.metric("Number of Nodes", len(region.mesh.points))
        st.metric("Number of Elements", len(cells))
        st.metric("Degrees of Freedom", displacement.values.size)

        st.divider()

        st.subheader("🎨 Visualization Info")
        st.info(f"""
        - **Red**: Maximum displacement
        - **Blue**: Minimum displacement
        - **Scale**: {scale_factor}x magnification
        - **Material**: {material_type}
        """)

else:
    st.info(
        "👈 Configure parameters in the sidebar and click **Solve** to run the "
        "simulation.",
    )

    # Show example image or description
    st.markdown("""
    ### About this example

    This simulation models a cantilever beam:
    - **Dimensions**: 2000mm x 100mm x 100mm
    - **Boundary Condition**: Fixed at x=0 (left end)
    - **Loading**: Gravity acting downward
    - **Analysis Type**: Linear elastic

    The beam will deflect under its own weight, with maximum displacement at the
    free end.
    """)
