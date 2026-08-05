# ML4HLS_Tutorial_base

Repository that contains the base information for make a machine learning inference in FPGA using HLS4ML

## Clonar el repositorio para incluir submodulos

Para clonar el repositorio con exito y sin errores usar la siguiente linea

```bash
git clone -b main --recurse-submodules https://github.com/fabioc9675/ML4HLS_Tutorial_base.git
```

Con `--recurse-submodules` se garantiza que se clonan los submodulos completos, y con `-b main` es para clonar el branch especifico,

### Agregar submodulos de udma, comblock

Submodulo de udma

```
git submodule add https://gitlab.com/ictp-mlab/udma
```

Submodulo de comblock

```
git submodule add https://gitlab.com/ictp-mlab/core-comblock
```
