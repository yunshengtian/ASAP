#include "Body/BodyMultiSDFObj.h"
#include "makelevelset3.h"

namespace redmax {

BodyMultiSDFObj::BodyMultiSDFObj(
    Simulation* sim, Joint* joint,
    std::vector<std::string> filenames,
    std::vector<Matrix3> Rs, std::vector<Vector3> ps, 
    dtype dx, int res, dtype col_th,
    TransformType transform_type,
    dtype density,
    Vector3 scale,
    bool adaptive_sample,
    bool load_sdf,
    bool save_sdf) 
    : BodyMultiMeshObj(sim, joint, filenames, Rs, ps, transform_type, density, scale, adaptive_sample) {

    bool sdf_loaded = false;
    if (load_sdf) {
        sdf_loaded = load_SDF();
    }
    if (!sdf_loaded) {
        precompute_SDF(dx, res);
        if (save_sdf) {
            save_SDF();
        }
    }

    // std::cerr << "Before computing SDF" << std::endl;
    // precompute_SDF(dx);
    _col_th = col_th;
    // std::cerr << "Finish computing SDF" << std::endl;
    // std::cerr << "SDF size: " << _SDF.ni << "," << _SDF.nj << "," << _SDF.nk << std::endl;
    // std::cerr << "SDF min box: (" << _min_box[0] << "," << _min_box[1] << "," << _min_box[2] << "), max box: (" << _max_box[0] << "," << _max_box[1] << "," << _max_box[2] << ")" << std::endl;
}

void BodyMultiSDFObj::precompute_SDF(dtype dx, int res) {

    _min_boxes.clear();
    _max_boxes.clear();
    _dxs.clear();
    _dys.clear();
    _dzs.clear();
    _SDFs.clear();

    for (int fi = 0; fi < _filenames.size(); ++fi) {

        //start with a massive inside out bound box.
        sdfgen::Vec3f min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()), 
            max_box(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

        std::vector<sdfgen::Vec3f> vertList;
        std::vector<sdfgen::Vec3ui> faceList;

        int num_vertices = _Vs[fi].cols();
        for (int i = 0;i < num_vertices;i++) {
            sdfgen::Vec3f point(_Vs[fi](0, i), _Vs[fi](1, i), _Vs[fi](2, i));
            vertList.push_back(point);
            sdfgen::update_minmax(point, min_box, max_box);
        }

        int num_faces = _Fs[fi].cols();
        for (int i = 0;i < num_faces;i++) {
            faceList.push_back(sdfgen::Vec3ui(_Fs[fi](0, i), _Fs[fi](1, i), _Fs[fi](2, i)));
        }

        int min_size = res;
        Vector3 bbox_size = _bounding_boxes[fi].second - _bounding_boxes[fi].first;
        // Vector3 bbox_size = _bounding_box.second - _bounding_boxes[fi].first;
        dtype _dx = std::min(dx, bbox_size(0) / min_size);
        dtype _dy = std::min(dx, bbox_size(1) / min_size);
        dtype _dz = std::min(dx, bbox_size(2) / min_size);
        _dxs.push_back(_dx);
        _dys.push_back(_dy);
        _dzs.push_back(_dz);

        // add padding around the box
        sdfgen::Vec3f unit((float)_dx, (float)_dy, (float)_dz);
        int padding = 1;
        min_box -= (float)padding * unit;
        max_box += (float)padding * unit;
        sdfgen::Vec3ui sizes = sdfgen::Vec3ui((max_box - min_box + sdfgen::Vec3f(0.5 * _dx, 0.5 * _dy, 0.5 * _dz)) / sdfgen::Vec3f(_dx, _dy, _dz)) + sdfgen::Vec3ui(1, 1, 1);

        // make SDF
        sdfgen::Array3f SDF;
        sdfgen::make_level_set3(faceList, vertList, min_box, (float)_dx, (float)_dy, (float)_dz, sizes[0], sizes[1], sizes[2], SDF);

        // type conversion
        Vector3 _min_box = Vector3(min_box[0], min_box[1], min_box[2]);
        Vector3 _max_box = Vector3(max_box[0], max_box[1], max_box[2]);
        _min_boxes.push_back(_min_box);
        _max_boxes.push_back(_max_box);

        int ni = SDF.ni, nj = SDF.nj, nk = SDF.nk;
        sdfgen::Array3<dtype> _SDF;
        _SDF.resize(ni, nj, nk);
        for (int i = 0; i < ni; ++i) for (int j = 0; j < nj; ++j) for (int k = 0; k < nk; ++k) {
            _SDF(i, j, k) = (dtype)SDF(i, j, k);
        }
        _SDFs.push_back(_SDF);
    }
}

bool BodyMultiSDFObj::load_SDF() {

    _min_boxes.clear();
    _max_boxes.clear();
    _dxs.clear();
    _dys.clear();
    _dzs.clear();
    _SDFs.clear();

    for (int fi = 0; fi < _filenames.size(); ++fi) {
        
        std::string inname = _filenames[fi].substr(0, _filenames[fi].size()-4) + std::string(".sdf");
        std::ifstream infile(inname.c_str());
        bool success = infile.good();
        if (success) {
            int ni, nj, nk;
            infile >> ni >> nj >> nk;
            Vector3 _min_box, _max_box;
            infile >> _min_box[0] >> _min_box[1] >> _min_box[2];
            infile >> _max_box[0] >> _max_box[1] >> _max_box[2];
            dtype _dx, _dy, _dz;
            infile >> _dx >> _dy >> _dz;
            sdfgen::Array3<dtype> _SDF;
            _SDF.resize(ni, nj, nk);
            for(unsigned int i = 0; i < _SDF.a.size(); ++i) {
                infile >> _SDF.a[i];
            }

            _min_boxes.push_back(_min_box);
            _max_boxes.push_back(_max_box);
            _dxs.push_back(_dx);
            _dys.push_back(_dy);
            _dzs.push_back(_dz);
            _SDFs.push_back(_SDF);
        }
        infile.close();
        if (!success) return false;
    }
    return true;
}

void BodyMultiSDFObj::save_SDF() {

    for (int fi = 0; fi < _filenames.size(); ++fi) {

        std::string outname = _filenames[fi].substr(0, _filenames[fi].size()-4) + std::string(".sdf");    
        std::ofstream outfile(outname.c_str());
        outfile << _SDFs[fi].ni << " " << _SDFs[fi].nj << " " << _SDFs[fi].nk << std::endl;
        outfile << _min_boxes[fi][0] << " " << _min_boxes[fi][1] << " " << _min_boxes[fi][2] << std::endl;
        outfile << _max_boxes[fi][0] << " " << _max_boxes[fi][1] << " " << _max_boxes[fi][2] << std::endl;
        outfile << _dxs[fi] << " " << _dys[fi] << " " << _dzs[fi] << std::endl;
        for(unsigned int i = 0; i < _SDFs[fi].a.size(); ++i) {
            outfile << _SDFs[fi].a[i] << std::endl;
        }
        outfile.close();
    }
}

void BodyMultiSDFObj::clear_saved_SDF() {

    for (int fi = 0; fi < _filenames.size(); ++fi) {
        std::string name = _filenames[fi].substr(0, _filenames[fi].size()-4) + std::string(".sdf");    
        remove(name.c_str());
    }
}

dtype BodyMultiSDFObj::query_SDF(int fi, Vector3 x, bool outside_accurate) {

    bool inside = true;
    for (int i = 0;i < 3;i++) {
        inside = inside && x(i) >= _min_boxes[fi][i] && x(i) <= _max_boxes[fi][i];
    }

    if (inside) {
        // compute grid location and interpolation coefficients
        dtype ci = (x(0) - _min_boxes[fi][0]) / _dxs[fi], cj = (x(1) - _min_boxes[fi][1]) / _dys[fi], ck = (x(2) - _min_boxes[fi][2]) / _dzs[fi];
        int i = std::floor(ci), j = std::floor(cj), k = std::floor(ck);
        ci = ci - i, cj = cj - j, ck = ck - k;

        // boundary handling
        int ni = _SDFs[fi].ni, nj = _SDFs[fi].nj, nk = _SDFs[fi].nk;
        int i_pos = (i == ni - 1) ? i : i + 1;
        int j_pos = (j == nj - 1) ? j : j + 1;
        int k_pos = (k == nk - 1) ? k : k + 1;

        // interpolate along i-axis
        dtype vi_j0k0 = (1 - ci) * _SDFs[fi](i, j, k) + ci * _SDFs[fi](i_pos, j, k);
        dtype vi_j1k0 = (1 - ci) * _SDFs[fi](i, j_pos, k) + ci * _SDFs[fi](i_pos, j_pos, k);
        dtype vi_j0k1 = (1 - ci) * _SDFs[fi](i, j, k_pos) + ci * _SDFs[fi](i_pos, j, k_pos);
        dtype vi_j1k1 = (1 - ci) * _SDFs[fi](i, j_pos, k_pos) + ci * _SDFs[fi](i_pos, j_pos, k_pos);

        // interpolate along j-axis
        dtype vij_k0 = (1 - cj) * vi_j0k0 + cj * vi_j1k0;
        dtype vij_k1 = (1 - cj) * vi_j0k1 + cj * vi_j1k1;

        // interpolate along k-axis
        dtype v = (1 - ck) * vij_k0 + ck * vij_k1;

        return v;

    } else {

        if (outside_accurate) {

            // find closest boundary point
            Vector3 x_b;
            for (int i = 0; i < 3; ++i) x_b(i) = std::min(std::max(x(i), _min_boxes[fi][i]), _max_boxes[fi][i]);

            // approximate by boundary distance + boundary SDF
            dtype v = (x - x_b).norm() + query_SDF(fi, x_b);
            return v;

        } else {
            return (dtype)std::numeric_limits<float>::max();
        }
    }

}

void BodyMultiSDFObj::query_SDFs(dtype& min_dist, int& min_fi, Vector3 x, bool outside_accurate) {
    min_dist = (dtype)std::numeric_limits<float>::max();
    min_fi = 0;
    for (int fi = 0; fi < _filenames.size(); ++fi) {
        dtype dist = query_SDF(fi, x, outside_accurate);
        if (dist < min_dist) {
            min_dist = dist;
            min_fi = fi;
        }
    }
}

Vector3 BodyMultiSDFObj::query_dSDF(int fi, Vector3 x) {

    dtype eps = 1e-5;
    Vector3 dx = Vector3(eps, 0, 0), dy = Vector3(0, eps, 0), dz = Vector3(0, 0, eps);
    Vector3 v;
    v(0) = query_SDF(fi, x + dx, true) - query_SDF(fi, x - dx, true);
    v(1) = query_SDF(fi, x + dy, true) - query_SDF(fi, x - dy, true);
    v(2) = query_SDF(fi, x + dz, true) - query_SDF(fi, x - dz, true);
    v /= (2 * eps);
    
    return v;
}

Matrix3 BodyMultiSDFObj::query_ddSDF(int fi, Vector3 x) {
    
    dtype eps = 1e-5;
    Vector3 dx = Vector3(eps, 0, 0), dy = Vector3(0, eps, 0), dz = Vector3(0, 0, eps);
    Vector3 halfdx = dx / 2, halfdy = dy / 2, halfdz = dz / 2;
    Matrix3 v = Matrix3::Zero();
    dtype x_SDF = query_SDF(fi, x, true);
    v(0, 0) = query_SDF(fi, x + dx, true) + query_SDF(fi, x - dx, true) - 2 * x_SDF;
    v(1, 1) = query_SDF(fi, x + dy, true) + query_SDF(fi, x - dy, true) - 2 * x_SDF;
    v(2, 2) = query_SDF(fi, x + dz, true) + query_SDF(fi, x - dz, true) - 2 * x_SDF;
    v(0, 1) = v(1, 0) = query_SDF(fi, x + halfdx + halfdy, true) + query_SDF(fi, x - halfdx - halfdy, true) - query_SDF(fi, x + halfdx - halfdy, true) - query_SDF(fi, x - halfdx + halfdy, true);
    v(0, 2) = v(2, 0) = query_SDF(fi, x + halfdx + halfdz, true) + query_SDF(fi, x - halfdx - halfdz, true) - query_SDF(fi, x + halfdx - halfdz, true) - query_SDF(fi, x - halfdx + halfdz, true);
    v(1, 2) = v(2, 1) = query_SDF(fi, x + halfdy + halfdz, true) + query_SDF(fi, x - halfdy - halfdz, true) - query_SDF(fi, x + halfdy - halfdz, true) - query_SDF(fi, x - halfdy + halfdz, true);
    v /= (eps * eps);

    return v;
}

dtype BodyMultiSDFObj::distance(Vector3 xw) {

    Matrix3 R2 = _E_0i.topLeftCorner(3, 3);
    Vector3 p2 = _E_0i.topRightCorner(3, 1);
    Vector3 x = R2.transpose() * (xw - p2);
    dtype dist;
    int fi;
    query_SDFs(dist, fi, x);
    return dist + _col_th;
}

Vector3 BodyMultiSDFObj::surface_contact_pt(Vector3 xw) {

    Matrix3 R2 = _E_0i.topLeftCorner(3, 3);
    Vector3 p2 = _E_0i.topRightCorner(3, 1);
    Vector3 x = R2.transpose() * (xw - p2);

    // compute d
    dtype dist;
    int fi;
    query_SDFs(dist, fi, x);
    dtype d = dist + _col_th;
    
    // compute e
    Vector3 dd_dx = query_dSDF(fi, x);
    dtype dd_dx_norm = dd_dx.norm();
    Vector3 e;
    if (std::abs(dd_dx_norm) < 1e-5) {
        e = Vector3::Zero();
    } else {
        e = dd_dx / dd_dx_norm;
    }

    // compute xi2
    Vector3 xi2 = x - d * e;
    return xi2;
}

void BodyMultiSDFObj::collision(
    Vector3 xw, Vector3 xw_dot, /* input */
    dtype &d, Vector3 &n,  /* output */
    dtype &ddot, Vector3 &tdot,
    Vector3 &xi2) {

    Matrix3 I = Matrix3::Identity();
    Matrix3 R2 = _E_0i.topLeftCorner(3, 3);
    Vector3 p2 = _E_0i.topRightCorner(3, 1);
    Vector3 w2_dot = _phi.head(3);
    Vector3 v2_dot = _phi.tail(3);
    Vector3 p2_dot = R2 * v2_dot;
    Vector3 x = R2.transpose() * (xw - p2);

    // compute d
    dtype dist;
    int fi;
    query_SDFs(dist, fi, x);
    d = dist + _col_th;

    // compute e
    Vector3 dd_dx = query_dSDF(fi, x);
    Vector3 e = dd_dx / dd_dx.norm();

    // compute n
    n = R2 * e;

    // compute xi2
    xi2 = x - d * e;

    // compute ddot
    Vector3 xdot = math::skew(w2_dot).transpose() * x + R2.transpose() * xw_dot - v2_dot;
    ddot = dd_dx.transpose() * xdot;
    
    // compute tdot
    Vector3 xw2_dot = R2 * (w2_dot.cross(xi2) + v2_dot);
    Vector3 vw = xw_dot - xw2_dot;
    tdot = (I - n * n.transpose()) * vw;
}

// refer to https://www.overleaf.com/project/5cd0a78e0e41fe0f8632a42a
// section 1.4: General Penalty-based Contact
void BodyMultiSDFObj::collision(
    Vector3 xw, Vector3 xw_dot, /* input */
    dtype &d, Vector3 &n,  /* output */
    dtype &ddot, Vector3 &tdot,
    Vector3 &xi2,
    RowVector3 &dd_dxw, RowVector6 &dd_dq2, /* derivatives for d */ 
    Matrix3 &dn_dxw, Matrix36 &dn_dq2, /* derivatives for n */
    RowVector3 &dddot_dxw, RowVector3 &dddot_dxwdot, /* derivatives for ddot */
    RowVector6 &dddot_dq2, RowVector6 &dddot_dphi2,
    Matrix3 &dtdot_dxw, Matrix3 &dtdot_dxwdot, /* derivatives for tdot */
    Matrix36 &dtdot_dq2, Matrix36 &dtdot_dphi2,
    Matrix3 &dxi2_dxw, Matrix36 &dxi2_dq2/* derivatives for xi2 */) {

    Matrix3 I = Matrix3::Identity();
    Matrix3 R2 = _E_0i.topLeftCorner(3, 3);
    Vector3 p2 = _E_0i.topRightCorner(3, 1);
    Vector3 w2_dot = _phi.head(3);
    Vector3 v2_dot = _phi.tail(3);
    Vector3 p2_dot = R2 * v2_dot;
    Vector3 x = R2.transpose() * (xw - p2);

    /**************** values ****************/

    dtype dist;
    int fi;
    query_SDFs(dist, fi, x);
    d = dist + _col_th;
    Vector3 dd_dx = query_dSDF(fi, x);
    Vector3 e = dd_dx / dd_dx.norm();
    n = R2 * e;
    xi2 = x - d * e;
    Vector3 xdot = math::skew(w2_dot).transpose() * x + R2.transpose() * xw_dot - v2_dot;
    ddot = dd_dx.transpose() * xdot;
    Vector3 xw2_dot = R2 * (w2_dot.cross(xi2) + v2_dot);
    Vector3 vw = xw_dot - xw2_dot;
    tdot = (I - n * n.transpose()) * vw;

    /**************** derivatives ****************/

    Vector3 e1 = Vector3::UnitX(), e2 = Vector3::UnitY(), e3 = Vector3::UnitZ();

    // x
    Matrix3 dx_dxw = R2.transpose();
    Matrix3 dx_dw2 = math::skew(x);
    Matrix3 dx_dv2 = -I;

    // d
    dd_dxw = dd_dx.transpose() * dx_dxw;
    dd_dq2.leftCols(3) = dd_dx.transpose() * dx_dw2;
    dd_dq2.rightCols(3) = dd_dx.transpose() * dx_dv2;

    // n
    Matrix3 de_ddd_dx = Matrix3::Zero();
    dtype dd_dx_norm = dd_dx.norm();
    for (int i = 0; i < 3; ++i) {
        Vector3 unit_vec = Vector3::Zero();
        unit_vec(i) = 1;
        de_ddd_dx.row(i) = (dd_dx_norm * unit_vec - dd_dx(i) * dd_dx / dd_dx_norm) / (dd_dx_norm * dd_dx_norm);
    }
    Matrix3 ddd_dx_dx = query_ddSDF(fi, x);
    Matrix3 de_dx = de_ddd_dx * ddd_dx_dx;
    Matrix3 de_dxw = de_dx * dx_dxw;
    Matrix3 de_dw2 = de_dx * dx_dw2;
    Matrix3 de_dv2 = de_dx * dx_dv2;
    dn_dxw = R2 * de_dxw;
    dn_dq2.leftCols(3) = R2 * de_dw2;
    dn_dq2.col(0) += R2 * math::skew(e1) * e;
    dn_dq2.col(1) += R2 * math::skew(e2) * e;
    dn_dq2.col(2) += R2 * math::skew(e3) * e;
    dn_dq2.rightCols(3) = R2 * de_dv2;

    // xi2
    dxi2_dxw = dx_dxw - e * dd_dxw - d * de_dxw;
    dxi2_dq2.leftCols(3) = dx_dw2 - e * dd_dq2.head(3) - d * de_dw2;
    dxi2_dq2.rightCols(3) = dx_dv2 - e * dd_dq2.tail(3) - d * de_dv2;

    // ddot
    dddot_dxw = (dd_dx.transpose() * math::skew(w2_dot).transpose() + xdot.transpose() * ddd_dx_dx) * R2.transpose();
    dddot_dxwdot = dd_dx.transpose() * R2.transpose();
    dddot_dq2.leftCols(3) = dd_dx.transpose() * (-math::skew(w2_dot) * math::skew(x) + math::skew(R2.transpose() * xw_dot)) + xdot.transpose() * ddd_dx_dx * dx_dw2;
    dddot_dq2.rightCols(3) = dd_dx.transpose() * math::skew(w2_dot) + xdot.transpose() * ddd_dx_dx * dx_dv2;
    dddot_dphi2.leftCols(3) = dd_dx.transpose() * math::skew(x);
    dddot_dphi2.rightCols(3) = dd_dx.transpose() * (-I);

    // xw2_dot
    Matrix3 dxw2dot_dxw = R2 * math::skew(w2_dot) * dxi2_dxw;
    Matrix3 dxw2dot_dxwdot = Matrix3::Zero();
    Matrix36 dxw2dot_dq2;
    dxw2dot_dq2.leftCols(3) = -R2 * math::skew(w2_dot.cross(xi2) + v2_dot) + R2 * math::skew(w2_dot) * dxi2_dq2.leftCols(3);
    dxw2dot_dq2.rightCols(3) = R2 * math::skew(w2_dot) * dxi2_dq2.rightCols(3);
    Matrix36 dxw2dot_dphi2;
    dxw2dot_dphi2.col(0) = R2 * math::skew(e1) * xi2;
    dxw2dot_dphi2.col(1) = R2 * math::skew(e2) * xi2;
    dxw2dot_dphi2.col(2) = R2 * math::skew(e3) * xi2;
    dxw2dot_dphi2.rightCols(3) = R2;

    // tdot
    dtdot_dxw = -dxw2dot_dxw - n.transpose() * vw * dn_dxw - n * vw.transpose() * dn_dxw + n * n.transpose() * dxw2dot_dxw;
    dtdot_dxwdot = (I - n * n.transpose()) * (I - dxw2dot_dxwdot);
    dtdot_dq2.leftCols(3) = -dxw2dot_dq2.leftCols(3) - n.transpose() * vw * dn_dq2.leftCols(3) - n * (vw.transpose() * dn_dq2.leftCols(3) - n.transpose() * dxw2dot_dq2.leftCols(3));
    dtdot_dq2.rightCols(3) = -dxw2dot_dq2.rightCols(3) - n.transpose() * vw * dn_dq2.rightCols(3) - n * (vw.transpose() * dn_dq2.rightCols(3) - n.transpose() * dxw2dot_dq2.rightCols(3));
    dtdot_dphi2 = (I - n * n.transpose()) * (- dxw2dot_dphi2);
}

Vector3i BodyMultiSDFObj::query_grid_location(int fi, Vector3 x) {
    dtype ci = (x(0) - _min_boxes[fi][0]) / _dxs[fi], cj = (x(1) - _min_boxes[fi][1]) / _dys[fi], ck = (x(2) - _min_boxes[fi][2]) / _dzs[fi];
    int i = std::floor(ci), j = std::floor(cj), k = std::floor(ck);
    return Vector3i(i, j, k);
}

void BodyMultiSDFObj::test_collision_derivatives() {
    srand(1000);
    // srand(time(0));
    dtype eps = 1e-7;
    // generate random xw, xw_dot, E_2, phi_2
    Eigen::Quaternion<dtype> quat_2(Vector4::Random());
    quat_2.normalize();
    Matrix4 E2 = Matrix4::Identity();
    E2.topLeftCorner(3, 3) = quat_2.toRotationMatrix();
    E2.topRightCorner(3, 1) = Vector3::Random();
    Vector6 phi2 = Vector6::Random();

    Vector3 x, xw1;
    Vector3 rnd_x;
    Vector3 _min_box = _min_boxes[0], _max_box = _max_boxes[0];
    dtype _dx = _dxs[0];
    Vector3 scale_x = _max_box - _min_box - 2 * Vector3::Ones() * _dx;
    Vector3 min_x = _min_box + Vector3::Ones() * _dx;

    Vector3 xw1_dot = Vector3::Random();

    while (true) {

        rnd_x = 0.5 * (Vector3::Random() + Vector3::Ones());

        x = rnd_x.cwiseProduct(scale_x) + min_x;
        xw1 = E2.topLeftCorner(3, 3) * x + E2.topRightCorner(3, 1);

        dtype dist;
        int fi;
        query_SDFs(dist, fi, x);
        if (dist >= 0.) {
            std::cerr << _min_box.transpose() << " " << _max_box.transpose() << std::endl;
            std::cerr << x.transpose() << std::endl;
            throw_error("Bug in test_collision_derivatives() in BodyMultiSDFObj.cpp");
        }

        Matrix3 R2 = E2.topLeftCorner(3, 3);
        Vector3 p2 = E2.topRightCorner(3, 1);

        std::vector<Vector3> xw_pos;
        xw_pos.push_back(xw1 + eps * xw1_dot);
        xw_pos.push_back(xw1 + eps * Vector3::UnitX() * eps);
        xw_pos.push_back(xw1 + eps * Vector3::UnitY() * eps);
        xw_pos.push_back(xw1 + eps * Vector3::UnitZ() * eps);

        std::vector<Vector3> x_pos;
        for (int i = 0; i < xw_pos.size(); ++i) {
            x_pos.push_back(R2.transpose() * (xw_pos[i] - p2));
        }
        
        Vector3i loc = query_grid_location(fi, x);
        std::vector<Vector3i> loc_delta;
        std::vector<Vector3> deltas;
        deltas.push_back(_dx / 2. * Vector3::UnitX());
        deltas.push_back(_dx / 2. * Vector3::UnitY());
        deltas.push_back(_dx / 2. * Vector3::UnitZ());
        for (int i = 0; i < deltas.size(); ++i) {
            loc_delta.push_back(query_grid_location(fi, x + deltas[i]));
        }

        for (int i = 0; i < x_pos.size(); ++i) {
            Vector3i loc_pos = query_grid_location(fi, x_pos[i]);
            if (!loc.isApprox(loc_pos)) {
                std::cerr << "loc and loc_pos: " << loc.transpose() << ", " << loc_pos.transpose() << std::endl;
                std::cerr << "position: " << x.transpose() << ", " << x_pos[i].transpose() << std::endl;
                throw_error("grid location mismatch");
            }
            for (int j = 0; j < deltas.size(); ++j) {
                Vector3i loc_pos_delta = query_grid_location(fi, x_pos[i] + deltas[j]);
                if (!loc_delta[j].isApprox(loc_pos_delta)) {
                    std::cerr << j << "th loc and loc_pos delta: " << loc_delta[j].transpose() << ", " << loc_pos_delta.transpose() << std::endl;
                    std::cerr << "position: " << (x + deltas[j]).transpose() << ", " << (x_pos[i] + deltas[j]).transpose() << std::endl;
                    throw_error("grid location mismatch");
                }
            }
        }

        break;
    }

    this->_E_0i = E2;
    this->_phi = phi2;

    // test SDF derivatives
    {
        Matrix3 R2 = E2.topLeftCorner(3, 3);
        Vector3 p2 = E2.topRightCorner(3, 1);

        dtype d_ori;
        int fi;
        query_SDFs(d_ori, fi, x);
        Vector3 dd_dx_ori = query_dSDF(fi, x);
        Vector3 dd_dx_fd;
        for (int i = 0;i < 3;i++) {
            Vector3 x_pos = x;
            x_pos[i] += eps;
            dtype d_pos = query_SDF(fi, x_pos);
            dd_dx_fd[i] = (d_pos - d_ori) / eps;
        }

        print_error("dd_dx", dd_dx_ori, dd_dx_fd);

        Matrix3 ddd_dx_ori = query_ddSDF(fi, x);
        Matrix3 ddd_dx_fd;
        for (int i = 0;i < 3;i++) {
            Vector3 x_pos = x;
            x_pos[i] += eps;
            Vector3 dd_dx_pos = query_dSDF(fi, x_pos);
            ddd_dx_fd.col(i) = (dd_dx_pos - dd_dx_ori) / eps;
        }

        print_error("ddd_dx", ddd_dx_ori, ddd_dx_fd);
    }

    // test time derivatives
    {
        dtype d_ori, ddot_ori;
        Vector3 n_ori, tdot_ori, xi2_ori;
        collision(xw1, xw1_dot, d_ori, n_ori, ddot_ori, tdot_ori, xi2_ori);

        dtype ddot_fd;

        Vector3 xw1_pos = xw1 + eps * xw1_dot;
        Vector6 dq = eps * phi2;
        Matrix4 E2_pos = E2 * math::exp(dq);
        this->_E_0i = E2_pos;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1_pos, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        ddot_fd = (d_pos - d_ori) / eps;
        this->_E_0i = E2;

        print_error("ddot", ddot_ori, ddot_fd);
    }

    dtype d, ddot;
    Vector3 n, tdot, xi2;
    RowVector3 dd_dxw1, dddot_dxw1, dddot_dxw1dot;
    RowVector6 dd_dq2, dddot_dq2, dddot_dphi2;
    Matrix3 dn_dxw1, dtdot_dxw1, dtdot_dxw1dot, dxi2_dxw1;
    Matrix36 dn_dq2, dtdot_dq2, dtdot_dphi2, dxi2_dq2;
    collision(xw1, xw1_dot, d, n, ddot, tdot, xi2,
                dd_dxw1, dd_dq2,
                dn_dxw1, dn_dq2,
                dddot_dxw1, dddot_dxw1dot,
                dddot_dq2, dddot_dphi2,
                dtdot_dxw1, dtdot_dxw1dot,
                dtdot_dq2, dtdot_dphi2,
                dxi2_dxw1, dxi2_dq2);

    // test dxw1 related
    RowVector3 dd_dxw1_fd, dddot_dxw1_fd;
    Matrix3 dn_dxw1_fd, dtdot_dxw1_fd, dxi2_dxw1_fd;
    for (int i = 0;i < 3;i++) {
        Vector3 xw1_pos = xw1;
        xw1_pos[i] += eps;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1_pos, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dd_dxw1_fd[i] = (d_pos - d) / eps;
        dddot_dxw1_fd[i] = (ddot_pos - ddot) / eps;
        dn_dxw1_fd.col(i) = (n_pos - n) / eps;
        dtdot_dxw1_fd.col(i) = (tdot_pos - tdot) / eps;
        dxi2_dxw1_fd.col(i) = (xi2_pos - xi2) / eps;
    }
    print_error("dd_dxw1", dd_dxw1, dd_dxw1_fd);
    print_error("dddot_dxw1", dddot_dxw1, dddot_dxw1_fd);
    print_error("dn_dxw1", dn_dxw1, dn_dxw1_fd);
    print_error("dtdot_dxw1", dtdot_dxw1, dtdot_dxw1_fd);
    print_error("dxi2_dxw1", dxi2_dxw1, dxi2_dxw1_fd);

    // test dxw1dot related
    RowVector3 dddot_dxw1dot_fd;
    Matrix3 dtdot_dxw1dot_fd;
    for (int i = 0;i < 3;i++) {
        Vector3 xw1dot_pos = xw1_dot;
        xw1dot_pos[i] += eps;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1, xw1dot_pos, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dddot_dxw1dot_fd[i] = (ddot_pos - ddot) / eps;
        dtdot_dxw1dot_fd.col(i) = (tdot_pos - tdot) / eps;
    }
    print_error("dddot_dxw1dot", dddot_dxw1dot, dddot_dxw1dot_fd);
    print_error("dtdot_dxw1dot", dtdot_dxw1dot, dtdot_dxw1dot_fd);

    // test dq2 related
    RowVector6 dd_dq2_fd, dddot_dq2_fd;
    Matrix36 dn_dq2_fd, dtdot_dq2_fd, dxi2_dq2_fd;
    for (int i = 0;i < 6;i++) {
        Vector6 dq = Vector6::Zero();
        dq[i] = eps;
        Matrix4 E2_pos = E2 * math::exp(dq);
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        this->_E_0i = E2_pos;
        collision(xw1, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dd_dq2_fd[i] = (d_pos - d) / eps;
        dddot_dq2_fd[i] = (ddot_pos - ddot) / eps;
        dn_dq2_fd.col(i) = (n_pos - n) / eps;
        dtdot_dq2_fd.col(i) = (tdot_pos - tdot) / eps;
        dxi2_dq2_fd.col(i) = (xi2_pos - xi2) / eps;
    }
    print_error("dd_dq2", dd_dq2, dd_dq2_fd);
    print_error("dddot_dq2", dddot_dq2, dddot_dq2_fd);
    print_error("dn_dw2", dn_dq2.leftCols(3), dn_dq2_fd.leftCols(3));
    print_error("dn_dv2", dn_dq2.rightCols(3), dn_dq2_fd.rightCols(3));
    print_error("dtdot_dw2", dtdot_dq2.leftCols(3), dtdot_dq2_fd.leftCols(3));
    print_error("dtdot_dv2", dtdot_dq2.rightCols(3), dtdot_dq2_fd.rightCols(3));
    print_error("dxi2_dq2", dxi2_dq2, dxi2_dq2_fd);
    this->_E_0i = E2;

    // test dphi2 related
    RowVector6 dddot_dphi2_fd;
    Matrix36 dtdot_dphi2_fd;
    for (int i = 0;i < 6;i++) {
        Vector6 phi2_pos = phi2;
        phi2_pos[i] += eps;
        this->_phi = phi2_pos;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dddot_dphi2_fd[i] = (ddot_pos - ddot) / eps;
        dtdot_dphi2_fd.col(i) = (tdot_pos - tdot) / eps;
    }
    print_error("dddot_dw2dot", dddot_dphi2.head(3), dddot_dphi2_fd.head(3));
    print_error("dddot_dv2dot", dddot_dphi2.tail(3), dddot_dphi2_fd.tail(3));
    print_error("dtdot_dw2dot", dtdot_dphi2.leftCols(3), dtdot_dphi2_fd.leftCols(3));
    print_error("dtdot_dv2dot", dtdot_dphi2.rightCols(3), dtdot_dphi2_fd.rightCols(3));
    this->_phi = phi2;
}

void BodyMultiSDFObj::test_collision_derivatives_runtime(Vector3 xw, Vector3 xw_dot) {
    dtype eps = 1e-7;

    Matrix4 E2 = this->_E_0i;
    Vector6 phi2 = this->_phi;
    Vector3 xw1 = xw;
    Vector3 xw1_dot = xw_dot;
    Vector3 x = E2.topLeftCorner(3, 3).transpose() * (xw1 - E2.topRightCorner(3, 1));

    dtype d, ddot;
    Vector3 n, tdot, xi2;
    RowVector3 dd_dxw1, dddot_dxw1, dddot_dxw1dot;
    RowVector6 dd_dq2, dddot_dq2, dddot_dphi2;
    Matrix3 dn_dxw1, dtdot_dxw1, dtdot_dxw1dot, dxi2_dxw1;
    Matrix36 dn_dq2, dtdot_dq2, dtdot_dphi2, dxi2_dq2;
    collision(xw1, xw1_dot, d, n, ddot, tdot, xi2,
                dd_dxw1, dd_dq2,
                dn_dxw1, dn_dq2,
                dddot_dxw1, dddot_dxw1dot,
                dddot_dq2, dddot_dphi2,
                dtdot_dxw1, dtdot_dxw1dot,
                dtdot_dq2, dtdot_dphi2,
                dxi2_dxw1, dxi2_dq2);

    // test time derivatives
    {
        // dtype d_ori, ddot_ori;
        // Vector3 n_ori, tdot_ori, xi2_ori;
        // collision(xw1, xw1_dot, d_ori, n_ori, ddot_ori, tdot_ori, xi2_ori);

        dtype ddot_fd;

        Vector3 xw1_pos = xw1 + eps * xw1_dot;
        Vector6 dq = eps * phi2;
        Matrix4 E2_pos = E2 * math::exp(dq);
        this->_E_0i = E2_pos;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1_pos, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
    
        ddot_fd = (d_pos - d) / eps;
        this->_E_0i = E2;

        print_error("Body SDF Collision Derivatives: ddot", ddot, ddot_fd);
    }

    // test dxw1 related
    RowVector3 dd_dxw1_fd, dddot_dxw1_fd;
    Matrix3 dn_dxw1_fd, dtdot_dxw1_fd, dxi2_dxw1_fd;
    for (int i = 0;i < 3;i++) {
        Vector3 xw1_pos = xw1;
        xw1_pos[i] += eps;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1_pos, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dd_dxw1_fd[i] = (d_pos - d) / eps;
        dddot_dxw1_fd[i] = (ddot_pos - ddot) / eps;
        dn_dxw1_fd.col(i) = (n_pos - n) / eps;
        dtdot_dxw1_fd.col(i) = (tdot_pos - tdot) / eps;
        dxi2_dxw1_fd.col(i) = (xi2_pos - xi2) / eps;
    }
    print_error("Body SDF Collision Derivatives: dd_dxw1", dd_dxw1, dd_dxw1_fd);
    print_error("Body SDF Collision Derivatives: dddot_dxw1", dddot_dxw1, dddot_dxw1_fd);
    print_error("Body SDF Collision Derivatives: dn_dxw1", dn_dxw1, dn_dxw1_fd);
    print_error("Body SDF Collision Derivatives: dtdot_dxw1", dtdot_dxw1, dtdot_dxw1_fd);
    print_error("Body SDF Collision Derivatives: dxi2_dxw1", dxi2_dxw1, dxi2_dxw1_fd);

    // test dxw1dot related
    RowVector3 dddot_dxw1dot_fd;
    Matrix3 dtdot_dxw1dot_fd;
    for (int i = 0;i < 3;i++) {
        Vector3 xw1dot_pos = xw1_dot;
        xw1dot_pos[i] += eps;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1, xw1dot_pos, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dddot_dxw1dot_fd[i] = (ddot_pos - ddot) / eps;
        dtdot_dxw1dot_fd.col(i) = (tdot_pos - tdot) / eps;
    }
    print_error("Body SDF Collision Derivatives: dddot_dxw1dot", dddot_dxw1dot, dddot_dxw1dot_fd);
    print_error("Body SDF Collision Derivatives: dtdot_dxw1dot", dtdot_dxw1dot, dtdot_dxw1dot_fd);

    // test dq2 related
    RowVector6 dd_dq2_fd, dddot_dq2_fd;
    Matrix36 dn_dq2_fd, dtdot_dq2_fd, dxi2_dq2_fd;
    for (int i = 0;i < 6;i++) {
        Vector6 dq = Vector6::Zero();
        dq[i] = eps;
        Matrix4 E2_pos = E2 * math::exp(dq);
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        this->_E_0i = E2_pos;
        collision(xw1, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dd_dq2_fd[i] = (d_pos - d) / eps;
        dddot_dq2_fd[i] = (ddot_pos - ddot) / eps;
        dn_dq2_fd.col(i) = (n_pos - n) / eps;
        dtdot_dq2_fd.col(i) = (tdot_pos - tdot) / eps;
        dxi2_dq2_fd.col(i) = (xi2_pos - xi2) / eps;
    }
    print_error("Body SDF Collision Derivatives: dd_dq2", dd_dq2, dd_dq2_fd);
    print_error("Body SDF Collision Derivatives: dddot_dq2", dddot_dq2, dddot_dq2_fd);
    print_error("Body SDF Collision Derivatives: dn_dw2", dn_dq2.leftCols(3), dn_dq2_fd.leftCols(3));
    print_error("Body SDF Collision Derivatives: dn_dv2", dn_dq2.rightCols(3), dn_dq2_fd.rightCols(3));
    print_error("Body SDF Collision Derivatives: dtdot_dw2", dtdot_dq2.leftCols(3), dtdot_dq2_fd.leftCols(3));
    print_error("Body SDF Collision Derivatives: dtdot_dv2", dtdot_dq2.rightCols(3), dtdot_dq2_fd.rightCols(3));
    print_error("Body SDF Collision Derivatives: dxi2_dq2", dxi2_dq2, dxi2_dq2_fd);
    this->_E_0i = E2;

    // test dphi2 related
    RowVector6 dddot_dphi2_fd;
    Matrix36 dtdot_dphi2_fd;
    for (int i = 0;i < 6;i++) {
        Vector6 phi2_pos = phi2;
        phi2_pos[i] += eps;
        this->_phi = phi2_pos;
        dtype d_pos, ddot_pos;
        Vector3 n_pos, tdot_pos, xi2_pos;
        collision(xw1, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
        dddot_dphi2_fd[i] = (ddot_pos - ddot) / eps;
        dtdot_dphi2_fd.col(i) = (tdot_pos - tdot) / eps;
    }
    print_error("Body SDF Collision Derivatives: dddot_dw2dot", dddot_dphi2.head(3), dddot_dphi2_fd.head(3));
    print_error("Body SDF Collision Derivatives: dddot_dv2dot", dddot_dphi2.tail(3), dddot_dphi2_fd.tail(3));
    print_error("Body SDF Collision Derivatives: dtdot_dw2dot", dtdot_dphi2.leftCols(3), dtdot_dphi2_fd.leftCols(3));
    print_error("Body SDF Collision Derivatives: dtdot_dv2dot", dtdot_dphi2.rightCols(3), dtdot_dphi2_fd.rightCols(3));
    this->_phi = phi2;
}

}
