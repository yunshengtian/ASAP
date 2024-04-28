#pragma once
#include "Body/BodyMultiMeshObj.h"
#include "array3.h"
#include "vec.h"

namespace redmax {

/**
 * BodySDFObj define a body loaded from an .obj file (triangle surface mesh) and compute a SDF for it.
 **/
class BodyMultiSDFObj : public BodyMultiMeshObj {
public:
    std::vector<Vector3> _min_boxes, _max_boxes;       // bounding box of SDF grid
    std::vector<dtype> _dxs, _dys, _dzs;              // delta of SDF grid
    dtype _col_th;                    // collision threshold
    std::vector<sdfgen::Array3<dtype>> _SDFs;       // voxel grid of SDF

    BodyMultiSDFObj(Simulation* sim, Joint* joint,
                    std::vector<std::string> filenames, 
                    std::vector<Matrix3> Rs, std::vector<Vector3> ps, 
                    dtype dx, int res, dtype col_th,
                    TransformType transform_type = BODY_TO_JOINT,
                    dtype density = (dtype)1.0,
                    Vector3 scale = Vector3::Ones(),
                    bool adaptive_sample = false,
                    bool load_sdf = false,
                    bool save_sdf = false);

    /**
     * analytical distance function
     * @param:
     * xw: the location of the query position in world frame
     * @return: contact distance of xw
    **/
    dtype distance(Vector3 xw);

    /**
     * get contact point on the surface
     */
    Vector3 surface_contact_pt(Vector3 xw);

    /**
     * analytical distance function return dist, normal, x2i, ddot, tdot
     * @return
     * dist: the distance of xw
     * normal: the normalized contact normal
     * xi2: the contact position on this body
     * ddot: the magnitude of the velocity on normal direction
     * tdot: the tangential velocity
     **/
    void collision(Vector3 xw, Vector3 xw_dot, /* input */
                    dtype &d, Vector3 &n,  /* output */
                    dtype &ddot, Vector3 &tdot,
                    Vector3 &xi2);

    /**
     * analytical distance function return dist, normal, x2i, ddot, tdot and derivatives
     * @return
     * dist: the distance of xw
     * normal: the normalized contact normal
     * xi2: the contact position on this body
     * ddot: the magnitude of the velocity on normal direction
     * tdot: the tangential velocity
     * derivatives
     **/
    void collision(Vector3 xw, Vector3 xw_dot, /* input */
                    dtype &d, Vector3 &n,  /* output */
                    dtype &ddot, Vector3 &tdot,
                    Vector3 &xi2,
                    RowVector3 &dd_dxw, RowVector6 &dd_dq2, /* derivatives for d */ 
                    Matrix3 &dn_dxw, Matrix36 &dn_dq2, /* derivatives for n */
                    RowVector3 &dddot_dxw, RowVector3 &dddot_dxwdot, /* derivatives for ddot */
                    RowVector6 &dddot_dq2, RowVector6 &dddot_dphi2,
                    Matrix3 &dtdot_dxw, Matrix3 &dtdot_dxwdot, /* derivatives for tdot */
                    Matrix36 &dtdot_dq2, Matrix36 &dtdot_dphi2,
                    Matrix3 &dxi2_dxw, Matrix36 &dxi2_dq2 /* derivatives for xi2 */);

    void test_collision_derivatives();
    void test_collision_derivatives_runtime(Vector3 xw, Vector3 xw_dot);

    void clear_saved_SDF();

private:
    void precompute_SDF(dtype dx, int res);
    bool load_SDF();
    void save_SDF();
    dtype query_SDF(int fi, Vector3 x, bool outside_accurate = true);
    void query_SDFs(dtype& min_dist, int& min_fi, Vector3 x, bool outside_accurate = true);
    Vector3 query_dSDF(int fi, Vector3 x);
    Matrix3 query_ddSDF(int fi, Vector3 x);
    Vector3i query_grid_location(int fi, Vector3 x);
};

}
