## Development
If you are using VSCode, install the recommended extensions and use Tasks instead of manually executing the commands.

#### Modifying it
The main entry point is the [debug executable](https://github.com/abc8747/am4/tree/master/src/am4/utils/cpp/main.cpp).

A C++17 compliant compiler and Linux system is required to build it.

!!! tip "Tip for VSCode users"
    Set build target and launch.

```sh
sudo apt-get install build-essential
# optionally install vtune for profiling

cd src/am4/utils
mkdir build
cd build
cmake .. && cmake --build . --target _core_executable && ./_core_executable
```
Note that the `BUILD_PYBIND` definition/directives controls whether the pybind11 bindings are included. It is set to 0 when building the executable.

#### Create the Python bindings
!!! tip "Tip for VSCode users"
    Run the `py: reinstall` task.

```sh
# in root dir:
sudo apt-get install python3-dev python3-pip
pip3 install virtualenv
virtualenv .venv
source .venv/bin/activate

pip3 install --verbose ".[dev,api,bot]" --config-settings=cmake.define.COPY_DATA=1
pytest
cd src/am4/utils
python3 generate-stubs.py
# pip3 uninstall src/am4/utils -y
```

The `am4` package and data files will then be installed in your site-packages, ready for use:
```py
from am4.utils.db import init
from am4.utils.aircraft import Aircraft
init() # IMPORTANT: loads the aircraft, airport and routes etc.
a = Aircraft.search("b744")
print(a.ac)
```
To learn more on how to use it, check out the [tests](https://github.com/abc8747/am4/tree/master/src/am4/utils/tests/), [generated stubs](https://github.com/abc8747/am4/tree/master/src/am4/utils/stubs/utils/) or the API reference.