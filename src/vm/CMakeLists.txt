add_library (
    vm
    OBJECT
        abs.c
)

qasm2m4_dep_on_c (abs.c sAbs)
