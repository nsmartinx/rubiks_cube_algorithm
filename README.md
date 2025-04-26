# rubiks_cube_algorithm

## Speed Test
Contains a few tests for brute forcing move sequences on cpu/gpu. The speed test 2 contains a few optimizations, including not repeating moves on the same facee, and containing hardcoded lookup tables for all of the moves.

### Building/Running
Run `make all` to build. The executables will be in `out`

## Algorithm_v1
Contains a modified verison of the Thistlethwaite Algorithm. Instead of the normal definition of G3, we use a stricter condition (as it is faster to check than the origional one), giving us a subgroup of G3.

### Building/Running
Run `make` to build. The executable will be in `out`

## Algorithm_v2
Generally the same approach as v1, but the code has been completely restructured to be more clean and maintainable.

### Building/Running
Run `./build` to build. The executable will be in `out`
