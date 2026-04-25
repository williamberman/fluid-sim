# fluid-sim

This is a simple (350 loc) fluid simulator for incompressible flows with constant kinematic viscosity on an uniformly spaced orthogonal mesh. It uses PISO for pressure correction.

### Cavity simulation

<video src="https://github.com/user-attachments/assets/5e1624b0-2be6-40a8-8876-22b57b94a2c5" controls width="600"></video>

left: this sim, right: OpenFOAM

The video is of the cell center and boundary velocity field.

The main differences are due to this sim using a constant upwind coefficient for the convection term for every face.

Cavity simulation produced by
```bash
python main.py --case_dir cavity --sim_dir cavity_sim
python visualize2d.py --case_dir cavity --sim_dir cavity_sim --vid_path cavity/cavity_sim.mp4
```

### TODOs

- I have only tested it on the openfoam cavity tutorial so its pretty likely its case directory parsing will break on other case directories.
- Proper upwind coefficients for the convection term. It just hard codes a constant value for all faces.
- uniformTotalPressure specified boundary conditions like in the TJunction tutorial.
- Non-uniform kinematic viscosity (also implement the second deviatoric stress term).
- Run on TJunction tutorial!
- The cavity mesh is small enough that we can just use scipy.sparse.linalg.spsolve. It could be cool to use some of the interative solver methods like in openfoam.

### Basic derivation

Somewhat docs, mainly useful for me sanity checking my implementation. Most of this was taken from reading the OpenFOAM source, so I'm sure there's a few mistakes.

#### Velocity solve from momentum equation

Dropping the density, $\rho$, at the beggining for brevity but the same derivation works using it and dropping it where appropriate.

Navier-stokes as integral form of time derivative of momentum for a continuum. Using the stress tensor for boundary forces in its separated hydrostatic and deviatoric components.

The $i$'th component of the momentum eq over a control volume $\Omega$. 

$\mathrm{dV} = \mathrm{d}x_0 \wedge \mathrm{d}x_1 \wedge \mathrm{d}x_2$

$\mathrm{dA}_i = \mathrm{d}x_j \wedge \mathrm{d}x_k$ where $i \neq j \neq k$ and $j < k$

```math
\mathrm{D}_t (\int_{\Omega} u_i \mathrm{dV}) = \int_{\Omega} a_i \mathrm{dV} + \int_{\mathrm{d}\Omega} -p \mathrm{dA}_i + \int_{\mathrm{d}\Omega} \sum_j \tau_{ij} \mathrm{dA}_j
```

Reynolds transport theorem

```math
\int_{\Omega} \mathrm{D}_t u_i \mathrm{dV} + \int_{\mathrm{d}\Omega} \sum_j u_i u_j \mathrm{dA}_j = \int_{\Omega} a_i \mathrm{dV} + \int_{\mathrm{d}\Omega} -p \mathrm{dA}_i + \int_{\mathrm{d}\Omega} \sum_j \tau_{ij} \mathrm{dA}_j
```

No body forces

```math
\int_{\Omega} \mathrm{D}_t u_i \mathrm{dV} + \int_{\mathrm{d}\Omega} \sum_j u_i u_j \mathrm{dA}_j = \int_{\mathrm{d}\Omega} -p \mathrm{dA}_i + \int_{\mathrm{d}\Omega} \sum_j \tau_{ij} \mathrm{dA}_j
```

Treat $u_j$ and $p$ as their previous time step values, $u_j'$ and $p'$. Maintains linearity and separation of the systems of the different spatial components.

```math
\int_{\Omega} \mathrm{D}_t u_i \mathrm{dV} + \int_{\mathrm{d}\Omega} u_i \sum_j u_j' \mathrm{dA}_j = \int_{\mathrm{d}\Omega} -p' \mathrm{dA}_i + \int_{\mathrm{d}\Omega} \sum_j \tau_{ij} \mathrm{dA}_j
```

$\sum_j u_j' \mathrm{dA}_j = \phi'$, the volumetric flux

```math
\int_{\Omega} \mathrm{D}_t u_i \mathrm{dV} + \int_{\mathrm{d}\Omega} u_i \phi' = \int_{\mathrm{d}\Omega} -p' \mathrm{dA}_i + \int_{\mathrm{d}\Omega} \sum_j \tau_{ij} \mathrm{dA}_j
```

Discretize integrals, with $^\mathrm{f}$ denoting values at the face.

```math
V \mathrm{D}_t u_i + \sum_{\mathrm{f}} u_i^\mathrm{f} \phi'^\mathrm{f} = \sum_{\mathrm{f}} -p'^{\mathrm{f}} \mathrm{dA}_i + \sum_{\mathrm{f}} \sum_j \tau_{ij}^\mathrm{f} \mathrm{dA}_j
```

Evaluate time derivative by finite differences

```math
\frac{V}{\Delta t} (u_i - u_i') + \sum_{\mathrm{f}} u_i^\mathrm{f} \phi'^\mathrm{f} = \sum_{\mathrm{f}} -p'^{\mathrm{f}} \mathrm{dA}_i + \sum_{\mathrm{f}} \sum_j \tau_{ij}^\mathrm{f} \mathrm{dA}_j
```

Evaluate deviatoric stress by constitutive equation. Under volume integral, for an incompressible fluid with constant kinematic viscosity, the second term is analytically zero, so we'll drop it.

```math
\tau_{ij} = \nu \mathrm{D}_{x_j} u_i + \nu \mathrm{D}_{x_i} u_j = \nu \mathrm{D}_{x_j} u_i
```

```math
\frac{V}{\Delta t} (u_i - u_i') + \sum_{\mathrm{f}} u_i^\mathrm{f} \phi'^\mathrm{f} = \sum_{\mathrm{f}} -p'^{\mathrm{f}} \mathrm{dA}_i + \nu \sum_{\mathrm{f}} \sum_j \mathrm{D}_{x_j} u_i^\mathrm{f} \mathrm{dA}_j
```

Now we choose methods for evaluating the field values and gradients at the faces. For the convection term, we weighted average the linear interpolation of the neighboring cell values and the upwind cell value. For the pressure term and the deviatoric stress term we linearly interpolate. We also use boundary specified values where appropriate. 

After face values have been specified in terms of cell values, we can form our three systems of equations for $u_0$, $u_1$, and $u_2$. 

#### Pressure correction from continuity equation

Our values for $u_i$ are not divergence free. We make them divergence free by solving the continuity equation for pressure and then updating the velocity field.

Rewrite our velocity system of equations by treating all diagonal $u_i$ terms as variables and all off diagonal $u_i$ terms as constants. Say $H_i$ has absorbed the constant terms and all the off diagonal $u_i$ terms multiplied by their most recent $u_i$ terms. Note that we have written the pressure term back in its differential form (i.e. un-integrated), this part is a bit handwavy to me because I feel like when you integrate the full equation to get the other coefficients, you shouldn't be allowed to choose a single term that is not integrated. However, I'm pretty sure this is what openfoam does.

```math
a_i u_i + H_i + D_{x_i} p = 0
```

```math
u_i = -\frac{H_i}{a_i} - \frac{1}{a_i} D_{x_i} p
```

We can substitute this into the continuity equation and integrate to get a system of equations for pressure.

```math
\sum_{i} D_{x_i} u_i = 0
```

```math
\sum_{i} D_{x_i} (-\frac{H_i}{a_i} - \frac{1}{a_i} D_{x_i} p) = 0
```

```math
\sum_{\mathrm{f}} \frac{1}{a_i} D_{x_i} p^\mathrm{f} = \sum_{\mathrm{f}} (\frac{H_i}{a_i})^\mathrm{f}
```

This solves for pressure which is then used to update the velocity field.

#### Misc

These formulas are presented for a single cell. We build the systems like openfoam does by the vectorized calculations of the face coefficients which are then added to cell terms in the sparse system matrix determined by face owner/neighbors. 
