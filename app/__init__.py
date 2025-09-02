import os
import sys
import subprocess
from glob import glob


# Directory where the Fortran files and compiled modules reside
APP_DIR = os.path.dirname(__file__)

# Define reasonable Fortran file extensions
FORTRAN_EXTENSIONS = [".f", ".f90", ".f95", ".for"]
COMPILED_DIRECTORY = "fortran"

# Directories for Fortran source files and compiled modules
BASE_DIR = os.path.dirname(APP_DIR)  # Project root
FORTRAN_DIR = os.path.join(BASE_DIR, "fortran")  # Directory with Fortran files
OUTPUT_DIR = os.path.join(APP_DIR, COMPILED_DIRECTORY)

def find_fortran_files(directory):
    """Finds all Fortran files in the specified directory."""
    fortran_files = []
    for ext in FORTRAN_EXTENSIONS:
        fortran_files.extend(glob(os.path.join(directory, f"*{ext}")))
    return fortran_files

def find_existing_compiled_file(module_name):
    """Checks if a compiled .so file for the given module already exists in compiled_modules."""
    so_pattern = os.path.join(OUTPUT_DIR, f"{module_name}.cpython-*.so")
    so_files = glob(so_pattern)
    return len(so_files) > 0  # Return True if at least one match is found

def ensure_init_py():
    """Ensures the __init__.py file exists in the compiled_modules directory."""
    init_file = os.path.join(OUTPUT_DIR, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            # Optionally write some metadata into __init__.py
            f.write("# This file makes 'compiled_modules' a Python package.\n")
        # print(f"Created {init_file}")

def compile_fortran_modules():
    """Compiles all Fortran files in the fortran directory into Python extension modules."""
    fortran_files = find_fortran_files(FORTRAN_DIR)
    
    if not fortran_files:
        print("No Fortran files found for compilation.")
        return

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_init_py()

    for fortran_file in fortran_files:
        module_name = os.path.splitext(os.path.basename(fortran_file))[0]

        # Skip compilation if a matching .so file already exists
        if find_existing_compiled_file(module_name):
            # print(f"Skipping {fortran_file} - compiled module already exists.")
            continue

        try:
            # print(f"Compiling {fortran_file} into module {module_name}...")
            subprocess.run(
                [
                    "python",
                    "-m",
                    "numpy.f2py",
                    "-c",
                    fortran_file,
                    "-m",
                    module_name,
                    "-llapack",
                    "-lblas",
                ],
                cwd=FORTRAN_DIR,  # Run the compilation command in the fortran directory
                check=True,
            )
            print("\n\n\n")
            # Find the generated .so file
            so_file = glob(os.path.join(FORTRAN_DIR, f"{module_name}.cpython-*.so"))
            if so_file:
                so_file = so_file[0]
                # Move the .so file to the output directory
                new_path = os.path.join(OUTPUT_DIR, os.path.basename(so_file))
                os.rename(so_file, new_path)
                # print(f"Moved {so_file} to {new_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile {fortran_file}: {e}")
            sys.exit(1)

# Compile Fortran modules automatically when the package is imported
# print("Checking and compiling Fortran modules...")
compile_fortran_modules()
# print("Fortran module setup complete.")

