# Mapel

Map of elections

# Installation

There are three major ways of using mapel:
1. Copy-paste the sources into your python project and import different `.py`
files directly;
2. (Recommended for developers) Install mapel as an editable package, which
simulates a full-fledged global package installation yet keeping track of all
changes to the source code and applying the on the fly to the "library";
3. (Recommended for library users) Install mapel as a normal package.

The first point is considered to be only used temporarily and creates severe
inconveniences with building the package. We only describe points 2 and 3 in
subsequent parts. For a better experience, our instructions assume usage of
`venv`, which is optional, however recommended.

For the full functionality of the package, it is recommended to also install extra
dependencies. Doing this is covered in parts describing respective installation
variants. The extra dependencies contain:  
```
cplex>=20.1.0.1
pulp~=2.5.1
abcvoting~=2.0.0b0
permanent
```  
which unlock approval based committee rules (which require solving I(L)P
programs) and sampling a matrix using a permanent-based approach.

> :exclamation: Note that this library contains C++ extensions. So using it
manually (point 1) requires a correct compilation of the `.cpp` files. How to
do this is beyond the scope of this instruction.

## Editable Package

This variant is especially recommended for developing purposes and for those
who would like to make changes in the code and have the changes reflected. The
instruction includes using `venv`, which is generally a good idea.

1. Install `pip` and `venv`. Make sure that you are using the newest possible
`pip`, newer than `22.0.0`. To upgrade you pip run:  
`pip install --upgrade pip'
2. Prepare a virtual environment running this command:  
`python3 -m venv <virtual_envirnonment_name>`
By default the above command creates a directory `<virtual_environment_name>`,
where the virtual environment files are stored
3. Activate the virtual environment:  
`source <virtual_envirnment_path>/bin/activate`  
If successful, your prompt is now preceded with the name of the virtual environment.
4. Clone the repository and go to its directory.
5. Run  
`pip install --editable .`  
to install the package and the necessary dependencies, also compiling all
necessary C++ extensions into python-readable libraries.
6. Run  
`pip install --editable .[extras]`  
to install extra dependencies that make the library more usable.

## Usual Package

This variant is recommended for those, who plan to use mapel without modifying
its source code. The instruction includes using `venv`, which is generally a
good idea.

All steps except from the 5th one are the same as in the Editable Package
variant. The new steps 5 and 6 are as follows:

5. Run  
`pip install .`  
1. Run  
`pip install .[extras]`  

> :exclamation: If you do not need the code and you want to have a version
 directly from the repository's `master` branch, then you can simply run: 
 `pip install -e  git+https://github.com/szufix/mapel#egg=mapel`



# Support

This project is part of the [PRAGMA project](https://home.agh.edu.pl/~pragma/)
which has received funding from the [European Research Council
(ERC)](https://home.agh.edu.pl/~pragma/) under the European Union’s Horizon 2020
research and innovation programme ([grant agreement No
101002854](https://erc.easme-web.eu/?p=101002854))



