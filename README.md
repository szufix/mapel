# Mapel

A collection of libraries allowing drawing various "maps of elections."

Besides the main package, `mapel-core`, the library currrently contains the
following plugins:
 - mapel-elections---handling maps of elections
 - mapel-marriages---handling maps of Stable Marriage instances
 - mapel-roommates---handling maos of Stable Roommates instances

# Installation

There are three major ways of using mapel:
1. Copy-paste the sources into your python project and import different `.py`
files directly;
2. (Recommended for developers) Install mapel, and the preferred plugins, as an
editable package, which simulates a full-fledged global package installation yet
keeping track of all changes to the source code and applying the on the fly to
the "library";
3. (Recommended for library users) Install mapel, and the preferred plugins, as
a normal packages.

The first point is considered to be only used temporarily and creates severe
inconveniences with building the package. We only describe points 2 and 3 in
subsequent parts. For a better experience, our instructions assume usage of
`venv`, which is optional, however recommended.

## Editable Package

This variant is especially recommended for developing purposes and for those
who would like to make changes in the code and have the changes reflected. The
instruction includes using `venv`, which is generally a good idea.

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
1. Clone the repository.
1. Go to directory `mapel-core` and run  
`pip install --editable .`  
to install the core of the library.
1. Repeat the above step for any other plugin you want to install; naturally,
using the correct directory.
1. For some of the libraries, you can run  
`pip install --editable .[extras]`  
to install extra dependencies that make the library more usable.


## Usual Package

This variant is recommended for those, who plan to use mapel without modifying
its source code. The instruction includes using `venv`, which is generally a
good idea.

To install the package as usuall, simply repeat steps 1, 2, and 3 from above.
Then run  
`pip install mapel-core`  
followed by installing further packages of your choice in the very same way.

## Testing Installation

TODO

# Support

This project is part of the [PRAGMA project](https://home.agh.edu.pl/~pragma/)
which has received funding from the [European Research Council
(ERC)](https://home.agh.edu.pl/~pragma/) under the European Union’s Horizon 2020
research and innovation programme ([grant agreement No
101002854](https://erc.easme-web.eu/?p=101002854))



