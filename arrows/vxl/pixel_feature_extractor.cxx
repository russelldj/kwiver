// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pixel_feature_extractor.h"

#include "aligned_edge_detection.h"
#include "average_frames.h"
#include "color_commonality_filter.h"
#include "convert_image.h"
#include "high_pass_filter.h"

#include <arrows/vxl/image_container.h>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>

#include <cstdlib>
#include <limits>
#include <type_traits>

namespace kwiver {

namespace arrows {

namespace vxl {

// ----------------------------------------------------------------------------
/// Private implementation class
class pixel_feature_extractor::priv
{
public:
  vxl::aligned_edge_detection aligned_edge_detection_filter;
  vxl::average_frames average_frames_filter;
  vxl::convert_image convert_image_filter;
  vxl::color_commonality_filter color_commonality_filter;
  vxl::high_pass_filter high_pass_filter;
};

// ----------------------------------------------------------------------------
pixel_feature_extractor
::pixel_feature_extractor()
  : d( new priv() )
{
  attach_logger( "arrows.vxl.pixel_feature_extractor" );
}

pixel_feature_extractor
::~pixel_feature_extractor()
{
}

vital::config_block_sptr
pixel_feature_extractor
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  return config;
}

// ----------------------------------------------------------------------------
void
pixel_feature_extractor
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed
  // values are present. An alternative is to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );
}

// ----------------------------------------------------------------------------
bool
pixel_feature_extractor
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
pixel_feature_extractor
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  auto aligned_edge = d->aligned_edge_detection_filter.filter( image_data );
  auto averaged = d->average_frames_filter.filter( image_data );
  auto converted = d->convert_image_filter.filter( image_data );
  auto color_commonality = d->color_commonality_filter.filter( image_data );
  auto high_pass = d->high_pass_filter.filter( image_data );

  std::vector< vil_image_view< vxl_byte > > filtered_images;
  filtered_images.push_back(
      vxl::image_container::vital_to_vxl( aligned_edge->get_image() ) );
  filtered_images.push_back(
      vxl::image_container::vital_to_vxl( averaged->get_image() ) );
  filtered_images.push_back(
      vxl::image_container::vital_to_vxl( converted->get_image() ) );
  filtered_images.push_back(
      vxl::image_container::vital_to_vxl( color_commonality->get_image() ) );
  filtered_images.push_back(
      vxl::image_container::vital_to_vxl( high_pass->get_image() ) );

  // Count the total number of planes
  size_t total_planes = 0;

  for( auto const& image : filtered_images )
  {
    total_planes += image.nplanes();
  }

  size_t ni = filtered_images.at( 0 ).ni();
  size_t nj = filtered_images.at( 0 ).nj();
  vil_image_view< vxl_byte > concatenated_out( ni, nj, total_planes );

  // Concatenate the filtered images into a single output
  size_t current_plane = 0;

  for( auto const& image : filtered_images )
  {
    for( size_t i = 0; i < image.nplanes(); ++i )
    {
      vil_plane( concatenated_out,
                 current_plane ).deep_copy( vil_plane( image, i ) );
      ++current_plane;
    }
  }

  vxl::image_container vxl_concatenated_out_container(
    concatenated_out );
  return std::make_shared< vxl::image_container >(
    vxl_concatenated_out_container );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
