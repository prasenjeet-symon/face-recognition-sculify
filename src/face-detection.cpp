
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------


frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;
anet_type net;


// ----------------------------------------------------------------------------------------

extern "C"
{

    EMSCRIPTEN_KEEPALIVE void init_shape_predictor(char input_buf[], uint32_t len)
    {
        std::string model(input_buf, len);
        std::istringstream model_istringstream(model);
        deserialize(sp, model_istringstream);
        cout << "Shape Predictor Inited." << endl;
        delete [] input_buf;
    }
}


// ----------------------------------------------------------------------------------------


extern "C"
{

    EMSCRIPTEN_KEEPALIVE void init_resnet_model(char input_buf[], uint32_t len)
    {
        std::string model(input_buf, len);
        std::istringstream model_istringstream(model);
        deserialize(net, model_istringstream);
        cout << "Resnet Model Inited." << endl;
        delete [] input_buf;
    }
}


// ----------------------------------------------------------------------------------------


extern "C"
{
    EMSCRIPTEN_KEEPALIVE uint32_t *recognize_face(unsigned char input_buf[], uint32_t width, uint32_t height)
    {
        matrix<rgb_pixel> input_image;
        input_image.set_size(height, width);

        // make the image matrix from the array
        for (int i = 0; i < height; i += 1)
        {
            for (int j = 0; j < width; j += 1)
            {
                uint32_t offset = (i * width * 4) + j * 4;
                unsigned char r = input_buf[offset];
                unsigned char g = input_buf[offset + 1];
                unsigned char b = input_buf[offset + 2];
                input_image(i, j) = {r, g, b};
            }
        }

        std::vector<matrix<rgb_pixel>> faces; // store all face chips found in the image
        std::vector<float> face_boxes; // store the face box info

        for (auto face : detector(input_image))
        {
            face_boxes.push_back(face.tl_corner().x());
            face_boxes.push_back(face.tl_corner().y());
            face_boxes.push_back(face.width());
            face_boxes.push_back(face.height());

            auto shape = sp(input_image, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(input_image, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
        }

        if (faces.size() != 0)
        {
            // find the face descriptors
            std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

            uint32_t *face_box_rectangles = new uint32_t[face_boxes.size() + 1];

            face_box_rectangles[0] = face_boxes.size() + 1;

            for (long i = 0; i < face_boxes.size(); i++)
            {
                face_box_rectangles[i + 1] = face_boxes[i];
            }

            return face_box_rectangles;
        }
        else
        {
            uint32_t *empty_face_box = new uint32_t[1];
            empty_face_box[0] = 1.0;
            return empty_face_box;
        }
    }
}
