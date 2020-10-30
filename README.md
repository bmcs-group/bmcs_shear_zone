
## Discrete cracking within a shear zone

The project implements a discrete crack propagation through 
a two dimensional domain cut out from a shear zone of a bended beam.  

[Crack propagation app](bmcs_shear/shear_crack/crack_propagation.ipynb)

The model has been developed by using a sketch 
in [notebooks/shear_zone.ipynb]([notebooks/shear_zone.ipynb])
which uses a combination of symbolic derivation supported by 
the `sympy` package and numerical `numpy` and `scipy` functionality.

Individual model components have been derived step by step 
indicating the mutual dependeces. The model is developed
in the following steps. 
 * beam design
   * material models (to be integrated with bmcs_beam)
 * crack path
   * crack tip rotation
 * deformed state
 * ligament stress profile
 * crack tip shear stress
 * crack orientation criterion
 * crack extension
 * crack propagation

The directory [shear_crack](bmcs_shear/shear_crack) demonstrates the 
applied development process. Each of the classes is implemented
in a separate Python file, i.e. `beam_design.py` and its functionality is demostrated
in the corresponding jupyter file, i.e. `beam_design.ipynb`.
