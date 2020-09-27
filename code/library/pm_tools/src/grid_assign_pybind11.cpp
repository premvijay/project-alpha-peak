// #include <iostream>

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <Python.h>

namespace py = pybind11;

// float W_ngp(float diff){
// return 1;
// };

float W_cic(float diff){
    float d = std::abs(diff);
    if (d < 1){
        return 1.0 - d;
    } else {
        return 0;
    };
};

float W_tsc(float diff){
    float d = std::abs(diff);
    if (d < 0.5){
        return (0.75 - d*d);
    } else if (d < 1.5){
        return 0.5 * std::pow(1.5 - d, 2);
    } else {
        return 0;
    };
};



size_t circ_index(float index, int N){
    if (index>=0 && index<N){ 
        return (size_t) index;
    } else if (index<0){
        return (size_t) (index + N);
    } else {
        return (size_t) (index - N);
    }
}


py::array_t<float> assign_ngp ( py::array_t<float> posd, int grid_size) {
    auto posd_unckd = posd.unchecked<2>();
    py::size_t num_prtcl = posd.shape(0);

    py::array_t<float> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = posd_unckd(i,0), yi = posd_unckd(i,1), zi = posd_unckd(i,2);
        float xif = std::floor(xi), yif = std::floor(yi), zif = std::floor(zi);

        size_t xc = circ_index(xif,grid_size), yc = circ_index(yif,grid_size), zc = circ_index(zif,grid_size);
        // float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

        grid_mut_unckd(xc,yc,zc) += 1;
    }
    return grid;
}


py::array_t<float> assign_cic ( py::array_t<float> posd, int grid_size) {
    auto posd_unckd = posd.unchecked<2>();
    py::size_t num_prtcl = posd.shape(0);

    py::array_t<float> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = posd_unckd(i,0), yi = posd_unckd(i,1), zi = posd_unckd(i,2);
        float xir = std::round(xi), yir = std::round(yi), zir = std::round(zi);

        for (int u=-1; u < 1; u++){
            for (int v=-1; v < 1; v++){
                for (int w=-1; w < 1; w++){
                    size_t xc = circ_index(xir+u, grid_size), yc = circ_index(yir+v, grid_size),
                                         zc = circ_index(zir+w, grid_size);
                    float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

                    grid_mut_unckd(xc,yc,zc) += W_cic(xcc-xi)* W_cic(ycc-yi)* W_cic(zcc-zi);
                }
            }
        }

        // grid_mut_unckd(xc,yc,zc)       += W_cic(xc-xi)*   W_cic(yc-yi)*   W_cic(zc-zi);
        // grid_mut_unckd(xc,yc,zc+1)     += W_cic(xc-xi)*   W_cic(yc-yi)*   W_cic(zc+1-zi);
        // grid_mut_unckd(xc,yc+1,zc)     += W_cic(xc-xi)*   W_cic(yc+1-yi)* W_cic(zc-zi);
        // grid_mut_unckd(xc,yc+1,zc+1)   += W_cic(xc-xi)*   W_cic(yc+1-yi)* W_cic(zc+1-zi);
        // grid_mut_unckd(xc+1,yc,zc)     += W_cic(xc+1-xi)* W_cic(yc-yi)*   W_cic(zc-zi);
        // grid_mut_unckd(xc+1,yc,zc+1)   += W_cic(xc+1-xi)* W_cic(yc-yi)*   W_cic(zc+1-zi);
        // grid_mut_unckd(xc+1,yc+1,zc)   += W_cic(xc+1-xi)* W_cic(yc+1-yi)* W_cic(zc-zi);
        // grid_mut_unckd(xc+1,yc+1,zc+1) += W_cic(xc+1-xi)* W_cic(yc+1-yi)* W_cic(zc+1-zi);
    }
    return grid;
}


py::array_t<float> assign_tsc ( py::array_t<float> posd, int grid_size) {
    auto posd_unckd = posd.unchecked<2>();
    py::size_t num_prtcl = posd.shape(0);

    py::array_t<float> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = posd_unckd(i,0), yi = posd_unckd(i,1), zi = posd_unckd(i,2);
        float xif = std::floor(xi), yif = std::floor(yi), zif = std::floor(zi);

        for (int u=-1; u < 2; u++){
            for (int v=-1; v < 2; v++){
                for (int w=-1; w < 2; w++){
                    size_t xc = circ_index(xif+u, grid_size), yc = circ_index(yif+v, grid_size),
                                         zc = circ_index(zif+w, grid_size);
                    float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

                    grid_mut_unckd(xc,yc,zc) += W_tsc(xcc-xi)* W_tsc(ycc-yi)* W_tsc(zcc-zi);
                }
            }
        }
    }
    return grid;
}




PYBIND11_MODULE(particles_to_grid, m) {
    m.doc() = "pybind11 based binding to C++ code for density assignment from particle positions based on CIC scheme. "; // optional module docstring

    m.def("assign_ngp", &assign_ngp, "Assign mass to grid from particle positions by nearest grid point scheme.");

    m.def("assign_cic", &assign_cic, "Assign mass to grid from particle positions by Cloud in cell scheme.");
            // py::arg("i"), py::arg("j")=0);

    m.def("assign_tsc", &assign_tsc, "Assign mass to grid from particle positions by triangularly shaped cloud scheme.");

    // m.def("arfn",&arfn, "pass an array");
}