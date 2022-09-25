# 07_observer_metamerism

## How to build the environment

### Windows 11

1. `pip install -r ./requirements.txt`
2. Add `../../ty_lib` to `PYTHONPATH`
3. Install OpenImageIO module (https://www.lfd.uci.edu/~gohlke/pythonlibs/#openimageio)

## Description of each file

* observer_metamerism.py
  * Implementation of simulation of color shift caused by the combination of the wide color gamut of the display device and observer metamerism.

* gamut_simulation.py
  * Implementation of simulation of the display gamut.

* spectrum.py
  * A library

* xyz_to_rgb_sample.py
  * A sample code for metamerism.

* ./def_data/
  * Spectral data
