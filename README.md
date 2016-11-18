# iceflow #
All the things

See the project wiki for more details on the workflow

## Setup ##
The following dependencies must be installed on your system to use these modules. Non-distributed libraries are included in this repo as submodules (see below)

### Python libraries ###
- GDAL/OGR
- NumPy

### Other ###
- [NASA Ames Stereo Pipeline (ASP)](https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/)

### Submodules ###
- pygeotools
- demcoreg

Everything should be setup on a mac (not tested on Windows) if you run
`make`
in your repo directory.

## Updates ##
If things change in submodules, you need to pull those changes too, so it is best if you always use

`git pull --recurse-submodules`

so that you get all changes with the pull.


