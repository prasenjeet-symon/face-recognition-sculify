
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

extern "C"
{
	EMSCRIPTEN_KEEPALIVE
		void init_shape_predictor(char input_buf[], uint32_t len)
		{
			std::string model(input_buf, len);
			std::istringstream model_istringstream(model);
			deserialize(sp, model_istringstream);
			cout << "Shape Predictor Inited." << endl;
			delete[] input_buf;
		}
}

extern "C"
{
	EMSCRIPTEN_KEEPALIVE
		void init_resnet_model(char input_buf[], uint32_t len)
		{
			std::string model(input_buf, len);
			std::istringstream model_istringstream(model);
			deserialize(net, model_istringstream);
			cout << "Resnet Model Inited." << endl;
			delete[] input_buf;
		}
}

extern "C"
{
	EMSCRIPTEN_KEEPALIVE
		uint16_t* recognize_face(unsigned char input_buf[], int width, int height)
		{
			// Image matrix - will store the image data
			matrix<rgb_pixel> img;
			img.set_size(height, width); // Set the image height and width

			// Make the image from the array
			for (int i = 0; i < height; i += 1)
			{
				for (int j = 0; j < width; j += 1)
				{
					uint32_t offset = (i * width * 4) + j * 4;
					unsigned char r = input_buf[offset];
					unsigned char g = input_buf[offset + 1];
					unsigned char b = input_buf[offset + 2];
					img(i, j) = {r, g, b};
				}
			}

			std::vector<matrix<rgb_pixel>> faces; // Will store the faces in the images
			std::vector<uint16_t> face_boxes;

			for (auto face : detector(img))
			{
				cout << "face detected " << endl;
				// Inert the face box
				face_boxes.push_back( face.tl_corner().x());
				face_boxes.push_back( face.tl_corner().y());
				face_boxes.push_back( face.width());
				face_boxes.push_back( face.height());

				auto shape = sp(img, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
				faces.push_back(move(face_chip));
			}

			if (faces.size() != 0)
			{
			    cout << "Face number is: " << faces.size() << endl;
				std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
			
				// matrix<float, 0, 1> single_person_face_descriptor = face_descriptors[0];
				// matrix<float, 1, 0> transposed_face_descriptor = trans(single_person_face_descriptor);
				// total number of row in the transposed_face_descriptor is equal to transposed_face_descriptor.nr()
				// create the memory on the heap with length equal to number of rows of transposed_face_descriptor

				// float* person_face_descriptor = new float[transposed_face_descriptor.nc() + 1];
				// person_face_descriptor[0] = transposed_face_descriptor.nc() + 1;
				// // loop through the matrix row and add the values to memory created
				// for ( long c = 0 ; c < transposed_face_descriptor.nc() ; ++c ) 
				// {
				// 	person_face_descriptor[c + 1] = transposed_face_descriptor(0, c);
				// }
				
				// // free the memory
				// delete[] person_face_descriptor;

			    // add the face boxes
				uint16_t* face_boxes_rectangles = new uint16_t[ ( faces.size() * 4 ) + 1];
				face_boxes_rectangles[0] = ( faces.size() * 4 ) + 1 ;

				for( long i = 0 ; i < face_boxes.size(); ++i )
				{
					face_boxes_rectangles[i + 1] = face_boxes[i];
				}

				return face_boxes_rectangles;

			}else
			{
				uint16_t* empty_descriptor = new uint16_t[1];
				cout << "No face detected " << endl;
				empty_descriptor[0] = 1;
				return empty_descriptor;

			}	
		}
}
