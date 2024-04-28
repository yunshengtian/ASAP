#pragma once
#include "Body/Body.h"

namespace redmax {

/**
 * BodyMeshObj define a body loaded from an .obj file (triangle surface mesh).
 **/
class BodyMultiMeshObj : public Body {
public:
    std::vector<std::string> _filenames;  // .obj filename (path)
    Matrix3X _V;            // vertices
    Matrix3Xi _F;            // face elements
    SE3 _E_oi;               // transform from body frame to obj frame
    SE3 _E_io;               // transform from obj frame to body frame
    int _sample_rate;       // sample rate of contact points

    std::vector<Vector3f> _colors;

    std::vector<Matrix3X> _Vs;
    std::vector<Matrix3Xi> _Fs;

    std::vector<std::pair<Vector3, Vector3>> _bounding_boxes; // bounding boxes of the body

    enum TransformType {
        BODY_TO_JOINT,
        OBJ_TO_WOLRD,
        OBJ_TO_JOINT,
        JOINT
    };

    BodyMultiMeshObj(Simulation* sim, Joint* joint,
                    std::vector<std::string> filenames, 
                    std::vector<Matrix3> Rs, std::vector<Vector3> ps, 
                    TransformType transform_type = BODY_TO_JOINT,
                    dtype density = (dtype)1.0,
                    Vector3 scale = Vector3::Ones(),
                    bool adaptive_sample = false);

    // mesh
    Matrix3X get_vertices() const;
    Matrix3Xi get_faces() const;

    // rendering
    void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list);

    bool filter_single(Vector3 xi);

    std::vector<int> filter(Matrix3X xi);

    std::vector<std::pair<Vector3, Vector3>> get_AABBs();

    // set rendering color
    void set_colors(std::vector<Vector3> colors);

private:
    void load_mesh(std::vector<std::string> filenames, std::vector<Matrix3> Rs, std::vector<Vector3> ps, Vector3 scale);

    Vector3 process_mesh();

    void compute_mass_property(const Matrix3X &V, const Matrix3Xi &F, /*input*/
                                dtype &mass, Vector3 &COM,          /*output*/
                                Matrix3 &I);
    
    // void compute_mass_property1(const Matrix3X &V, const Matrix3i &F, /*input*/
    //                             dtype &mass, Vector3 &COM,          /*output*/
    //                             Matrix3 &I);

    void precompute_bounding_box();

    void precompute_contact_points(bool adaptive_sample);
};

}