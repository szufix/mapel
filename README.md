# Mapel

The repo of mapel and its plugins containing the code of
the mapel ecosystem:  
1. [mapel-core](https://pypi.org/project/mapel-core/)
1. [mapel-elections](https://pypi.org/project/mapel-elections/)
1. [mapel-roommates](https://pypi.org/project/mapel-rommmates/)
1. [mapel-marriages](https://pypi.org/project/mapel-marriages/)

See the documentation of the above (on pypi) for further information on the
whole ecosystem.

To install all packages listed above follow the instructions below.


# Installation

There are three major ways of using mapel:
1. Copy-paste the sources into your python project and import different `.py`
files directly;
2. (Recommended for developers) [Documentation---TBD] Install mapel as an editable package, which
simulates a full-fledged global package installation yet keeping track of all
changes to the source code and applying the on the fly to the "library";
3. (Recommended for library users) Install mapel as a normal package.

The first point is considered to be only used temporarily and creates severe
inconveniences with building the package. We only describe points 2 (not yet)
and 3 in subsequent parts. For a better experience, our instructions assume
usage of `venv`, which is optional, however recommended.

## Editable Package

TBD

## Usual Package

This variant is recommended for those, who plan to use mapel without modifying
its source code. The instruction includes using `venv`, which is generally a
good idea.

1. Install `pip` and `venv`. Make sure that you are using the newest possible
`pip`, newer than `22.0.0`. To upgrade you pip run:
`pip install --upgrade pip`
1. Prepare a virtual environment running this command:
`python3 -m venv <virtual_envirnonment_name>`
By default the above command creates a directory `<virtual_environment_name>`,
where the virtual environment files are stored
1. Activate the virtual environment:
`source <virtual_envirnment_path>/bin/activate`
If successful, your prompt is now preceded with the name of the virtual environment.
1. Run
`pip install mapel`


# Acknowledgments

This project is part of the [PRAGMA project](https://home.agh.edu.pl/~pragma/)
which has received funding from the [European Research Council
(ERC)](https://home.agh.edu.pl/~pragma/) under the European Union’s Horizon 2020
research and innovation programme ([grant agreement No
101002854](https://erc.easme-web.eu/?p=101002854)).



