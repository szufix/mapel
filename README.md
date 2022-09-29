# Mapel

Map of elections



# Initial Setup

There are three major ways of using mapel:
1. Copy-paste the sources into your python project and import different `.py`
files directly;
2. Install mapel as an editable package, which simulates a full-fledged global
package installation yet keeping track of all changes to the source code and
applying the on the fly to the "library";
3. Install mapel as a normal package.

Currently, the best choice is point 2, which is subsequently described in this
readme. For a better experience, it is encuraged to use `venv`.

> :exclamation: Note that this library contains C++ extensions. So using it
manually (point 1) requires a correct compilation of the `.cpp` files. How to
do this is beyond the scope of this instruction.


## Editable Package

This variant is especially recomennded for developing purposes and for those
who would like to make changes in the code and have the changes reflected. Our
steps include using `venv`.

1. Install `pip`, `venv`, `boost` (in fact, one only needs the `boost_python`
library, so if you know what you are doing, go on and install only this
library).
2. Prepare a virtual environment running this command:  
`python3 -m venv <virtual_envirnonment_name>`
By default the above command creates a directory `<virtual_environment_name>`,
where the virtual environment files are stored
3. Activate the virtual environment:  
`source <virtual_envirnment_path>/bin/activate`  
If successful, your prompt is now preceeded with the name of the virtual envirnment.
4. Clone the repository and go its directory.
5. Run  
`pip install --editable .`  
This command installs the package, also compiling all necessary C++ extensions
into python-readable libraries.



# Support

This project is part of the [PRAGMA project](https://home.agh.edu.pl/~pragma/)
which has received funding from the [European Research Council
(ERC)](https://home.agh.edu.pl/~pragma/) under the European Union’s Horizon 2020
research and innovation programme ([grant agreement No
101002854](https://erc.easme-web.eu/?p=101002854))



