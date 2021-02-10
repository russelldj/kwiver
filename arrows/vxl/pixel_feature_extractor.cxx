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
// Private implementation class
class pixel_feature_extractor::priv
{
public:
  priv( pixel_feature_extractor* parent ) : p{ parent }
  {
  }

  // Copy multiple filtered images into contigious memory
  template< typename pix_t >
  vil_image_view< pix_t >
  concatenate_images( std::vector< vil_image_view< pix_t > > filtered_images );
  // Extract local pixel-wise features
  template < typename response_t >
  vil_image_view< response_t >
  filter( kwiver::vital::image_container_sptr input_image );

  pixel_feature_extractor* p;

  vxl::aligned_edge_detection aligned_edge_detection_filter;
  vxl::average_frames average_frames_filter;
  vxl::convert_image convert_image_filter;
  vxl::color_commonality_filter color_commonality_filter;
  // TODO add the edge filter
  vxl::high_pass_filter high_pass_filter;
};

// ----------------------------------------------------------------------------
template< typename pix_t >
vil_image_view< pix_t >
pixel_feature_extractor::priv
::concatenate_images( std::vector< vil_image_view< pix_t > > filtered_images )
{
  // Count the total number of planes
  unsigned total_planes{ 0 };
  for( auto const& image : filtered_images )
  {
    total_planes += image.nplanes();
  }

  if( total_planes == 0 )
  {
    LOG_ERROR( p->logger(), "No filtered images provided" );
    return {};
  }

  auto const ni = filtered_images.at( 0 ).ni();
  auto const nj = filtered_images.at( 0 ).nj();
  vil_image_view< pix_t > concatenated_out{ ni, nj, total_planes };

  // Concatenate the filtered images into a single output
  unsigned current_plane = 0;

  for( auto const& image : filtered_images )
  {
    for( unsigned i{ 0 }; i < image.nplanes(); ++i )
    {
      vil_plane( concatenated_out,
                 current_plane ).deep_copy( vil_plane( image, i ) );
      ++current_plane;
    }
  }
  return concatenated_out;
}

template < typename pix_t >
vil_image_view< pix_t >
pixel_feature_extractor::priv
::filter( kwiver::vital::image_container_sptr input_image )
{
  auto aligned_edge = aligned_edge_detection_filter.filter( input_image );
  auto averaged = average_frames_filter.filter( input_image );
  auto converted = convert_image_filter.filter( input_image );
  auto color_commonality = color_commonality_filter.filter( input_image );
  auto high_pass = high_pass_filter.filter( input_image );

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

  vil_image_view< vxl_byte > concatenated_out =
    concatenate_images< vxl_byte >( filtered_images );

  return concatenated_out;
}

// ----------------------------------------------------------------------------
pixel_feature_extractor
::pixel_feature_extractor()
  : d{ new priv{ this } }
{
  attach_logger( "arrows.vxl.pixel_feature_extractor" );
}

// ----------------------------------------------------------------------------
pixel_feature_extractor
::~pixel_feature_extractor()
{
}

// ----------------------------------------------------------------------------
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
  // Start with our generated vital::config_block to ensure that assumed values
  // are present. An alternative would be to check for key presence before
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
::filter( kwiver::vital::image_container_sptr image )
{
  // Perform Basic Validation
  if( !image )
  {
    LOG_ERROR( logger(), "Invalid image" );
    return kwiver::vital::image_container_sptr();
  }

  auto concatenated_responses = d->filter< vxl_byte >( image );

  vxl::image_container vxl_concatenated_out_container(
    concatenated_responses );
  return std::make_shared< vxl::image_container >(
    vxl_concatenated_out_container );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
