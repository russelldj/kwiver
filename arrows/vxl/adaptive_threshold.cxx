// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "adaptive_threshold.h"

#include "image_statistics.h"

#include <arrows/vxl/image_container.h>
#include <vital/range/iota.h>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>

#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

namespace kwiver {

namespace arrows {

namespace vxl {

// ----------------------------------------------------------------------------
// Private implementation class
class adaptive_threshold::priv
{
public:
  priv() = default;

  double fraction;
};

// ----------------------------------------------------------------------------
adaptive_threshold
::adaptive_threshold()
  : d{ new priv{} }
{
  attach_logger( "arrows.vxl.adaptive_threshold" );
}

// ----------------------------------------------------------------------------
adaptive_threshold
::~adaptive_threshold()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
adaptive_threshold
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "fraction", d->fraction,
                     "Upper fraction of pixels to mark as true." );

  return config;
}

// ----------------------------------------------------------------------------
void
adaptive_threshold
::set_configuration( vital::config_block_sptr in_config )
{
  // Start with our generated vital::config_block to ensure that assumed values
  // are present. An alternative would be to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->fraction = config->get_value< double >( "fraction" );
}

// ----------------------------------------------------------------------------
bool
adaptive_threshold
::check_configuration( vital::config_block_sptr in_config ) const
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  auto const fraction = config->get_value< double >( "fraction" );
  if( fraction < 0.0 || fraction > 1.0 )
  {
    LOG_ERROR( logger(), "fraction must be in [0, 1] but instead was "
               << fraction );
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
adaptive_threshold
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_ERROR( logger(), "Invalid image data." );
    return nullptr;
  }
  std::cout << "Filtering" << std::endl;

  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

#define HANDLE_CASE( T )                                                    \
  case T:                                                                   \
  {                                                                         \
    using ipix_t = vil_pixel_format_type_of< T >::component_type;           \
    vil_image_view< bool > thresholded;                                     \
    percentile_threshold_above< ipix_t >( view, d->fraction, thresholded ); \
    return std::make_shared< vxl::image_container>( thresholded );          \
  }

  return nullptr;

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_SBYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_64 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_64 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_FLOAT );
    HANDLE_CASE( VIL_PIXEL_FORMAT_DOUBLE );
#undef HANDLE_CASE

    default:
      LOG_ERROR( logger(), "Unsuported pixel type" );
      return nullptr;
  }
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
