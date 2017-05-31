/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/texture_extraction.hpp
 *
 * Copyright 2014-2017 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef XYZ_EXTRACTION_HPP_
#define XYZ_EXTRACTION_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/texture_extraction_detail.hpp"
#include "eos/render/render_affine.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "eos/render/utils.hpp" // for clip_to_screen_space()
#include "eos/render/Rasterizer.hpp"
#include "eos/render/FragmentShader.hpp"
#include "eos/fitting/closest_edge_fitting.hpp" // for ray_triangle_intersect()

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/glm.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <tuple>
#include <cassert>
#include <future>
#include <vector>

namespace eos {
	namespace render {

// Forward declarations:
//cv::Mat extract_xyz_isomap(core::Mesh mesh, int isomap_resolution = 512);

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 * This function can be used if a depth buffer has already been computed.
 * To just run the texture extraction, see the overload
 * extract_texture(Mesh, cv::Mat, cv::Mat, TextureInterpolation, int).
 *
 * It might be wise to remove this overload as it can get quite confusing
 * with the zbuffer. Obviously the depthbuffer given should have been created
 * with the same (affine or ortho) projection matrix than the texture extraction is called with.
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from.
 * @param[in] depthbuffer A pre-calculated depthbuffer image.
 * @param[in] compute_view_angle A flag whether the view angle of each vertex should be computed and returned. If set to true, the angle will be encoded into the alpha channel (0 meaning occluded or facing away 90°, 127 meaning facing a 45° angle and 255 meaning front-facing, and all values in between). If set to false, the alpha channel will only contain 0 for occluded vertices and 255 for visible vertices.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline cv::Mat extract_xyz_isomap(core::Mesh mesh, int isomap_resolution = 512)
{
	assert(mesh.vertices.size() == mesh.texcoords.size());

	using cv::Mat;
	using cv::Vec2f;
	using cv::Vec3f;
	using cv::Vec4f;
	using cv::Vec3b;
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;


    Mat isomap = Mat::zeros(isomap_resolution, isomap_resolution, CV_8UC3);
	// #Todo: We should handle gray images, but output a 4-channel isomap nevertheless I think.

	std::vector<std::future<void>> results;
	for (const auto& triangle_indices : mesh.tvi) {

		// Note: If there's a performance problem, there's no need to capture the whole mesh - we could capture only the three required vertices with their texcoords.
        auto extract_triangle = [&mesh, &triangle_indices, &isomap]() {

            //cv::Vec4f v0_as_Vec4f(mesh.vertices[triangle_indices[0]].x, mesh.vertices[triangle_indices[0]].y, mesh.vertices[triangle_indices[0]].z, mesh.vertices[triangle_indices[0]].w);
            //cv::Vec4f v1_as_Vec4f(mesh.vertices[triangle_indices[1]].x, mesh.vertices[triangle_indices[1]].y, mesh.vertices[triangle_indices[1]].z, mesh.vertices[triangle_indices[1]].w);
            //cv::Vec4f v2_as_Vec4f(mesh.vertices[triangle_indices[2]].x, mesh.vertices[triangle_indices[2]].y, mesh.vertices[triangle_indices[2]].z, mesh.vertices[triangle_indices[2]].w);
            cv::Vec3f v0_as_Vec3f(mesh.vertices[triangle_indices[0]].x, mesh.vertices[triangle_indices[0]].y, mesh.vertices[triangle_indices[0]].z);
            cv::Vec3f v1_as_Vec3f(mesh.vertices[triangle_indices[1]].x, mesh.vertices[triangle_indices[1]].y, mesh.vertices[triangle_indices[1]].z);
            cv::Vec3f v2_as_Vec3f(mesh.vertices[triangle_indices[2]].x, mesh.vertices[triangle_indices[2]].y, mesh.vertices[triangle_indices[2]].z);

            // The triangle in the isomap we want to fill
			cv::Point2f dst_tri[3];
            dst_tri[0] = cv::Point2f((isomap.cols - 0.5f)*mesh.texcoords[triangle_indices[0]][0], (isomap.rows - 0.5f)*mesh.texcoords[triangle_indices[0]][1] );
            dst_tri[1] = cv::Point2f((isomap.cols - 0.5f)*mesh.texcoords[triangle_indices[1]][0], (isomap.rows - 0.5f)*mesh.texcoords[triangle_indices[1]][1] );
            dst_tri[2] = cv::Point2f((isomap.cols - 0.5f)*mesh.texcoords[triangle_indices[2]][0], (isomap.rows - 0.5f)*mesh.texcoords[triangle_indices[2]][1] );

			// We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
			for (int x = min(dst_tri[0].x, min(dst_tri[1].x, dst_tri[2].x)); x < max(dst_tri[0].x, max(dst_tri[1].x, dst_tri[2].x)); ++x) {
				for (int y = min(dst_tri[0].y, min(dst_tri[1].y, dst_tri[2].y)); y < max(dst_tri[0].y, max(dst_tri[1].y, dst_tri[2].y)); ++y) {
					if (detail::is_point_in_triangle(cv::Point2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])) {

						// As the coordinates of the transformed pixel in the image will most likely not lie on a texel, we have to choose how to
						// calculate the pixel colors depending on the next texels
						// there are three different texture interpolation methods: area, bilinear and nearest neighbour

                        // Bilinear mapping: calculate pixel color depending on the distance to the triangle corners

                        // calculate euclidean distances to dst_tri points
                        using std::sqrt;
                        using std::pow;
                        float distance_v0 = sqrt(pow(x - dst_tri[0].x, 2) + pow(y - dst_tri[0].y, 2));
                        float distance_v1 = sqrt(pow(x - dst_tri[1].x, 2) + pow(y - dst_tri[1].y, 2));
                        float distance_v2 = sqrt(pow(x - dst_tri[2].x, 2) + pow(y - dst_tri[2].y, 2));

                        // normalise distances that the sum of all distances is 1
                        float sum_distances = distance_v0 + distance_v1 + distance_v2;
                        distance_v0 /= sum_distances;
                        distance_v1 /= sum_distances;
                        distance_v2 /= sum_distances;

                        // set xyz depending on distance from next 4 texels
                        Vec3f xyz_v0 = v0_as_Vec3f * distance_v0;
                        Vec3f xyz_v1 = v1_as_Vec3f * distance_v1;
                        Vec3f xyz_v2 = v2_as_Vec3f * distance_v2;

                        //isomap.at<Vec3b>(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
                        isomap.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(glm::clamp(xyz_v0[0] + xyz_v1[0] + xyz_v2[0]+128,0.f,255.0f));
                        isomap.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(glm::clamp(xyz_v0[1] + xyz_v1[1] + xyz_v2[1]+128,0.f,255.0f));
                        isomap.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(glm::clamp(xyz_v0[2] + xyz_v1[2] + xyz_v2[2]+128,0.f,255.0f));

					}
				}
			}
		}; // end lambda auto extract_triangle();
		results.emplace_back(std::async(extract_triangle));
	} // end for all mesh.tvi
	// Collect all the launched tasks:
	for (auto&& r : results) {
		r.get();
	}

	return isomap;
};


	} /* namespace render */
} /* namespace eos */

#endif /* XYZ_EXTRACTION_HPP_ */
