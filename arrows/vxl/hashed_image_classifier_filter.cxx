// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "hashed_image_classifier_filter.h"
#include "hashed_image_classifier.h"

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
class hashed_image_classifier_filter::priv
{
public:
  // Convert the type
  template < typename ipix_t > vil_image_view< ipix_t >
  convert( vil_image_view_base_sptr& view );

  // Scale and convert the image
  template < typename ipix_t, typename opix_t > vil_image_view< opix_t >
  scale( vil_image_view< ipix_t > input );

  vidtk::hashed_image_classifier< uint16_t, double > classifier;

  bool use_variable_models = false;
  float lower_gsd_threshold = 0.11;
  float upper_gsd_threshold = 0.22;
  std::string default_filename = "";
  std::string eo_narrow_filename = "";
  std::string eo_medium_filename = "";
  std::string eo_wide_filename = "";
  std::string ir_narrow_filename = "";
  std::string ir_medium_filename = "";
  std::string ir_wide_filename = "";
};

// ----------------------------------------------------------------------------
hashed_image_classifier_filter
::hashed_image_classifier_filter()
  : d( new priv() )
{
  attach_logger( "arrows.vxl.hashed_image_classifier_filter" );
}

hashed_image_classifier_filter
::~hashed_image_classifier_filter()
{
}

vital::config_block_sptr
hashed_image_classifier_filter
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "use_variable_models", d->use_variable_models,
                     "Set to true if we should use different models "
                     "for different GSDs and modalities." );
  config->set_value( "lower_gsd_threshold", d->lower_gsd_threshold,
                     "GSD threshold seperating the lowest from middle "
                     "GSD intervals used with variable model selection." );
  config->set_value( "upper_gsd_threshold", d->upper_gsd_threshold,
                     "GSD threshold seperating the middle from highest "
                     "GSD intervals used with variable model selection." );
  config->set_value( "default_filename", d->default_filename,
                     "Filename for the default model to use." );
  config->set_value( "eo_narrow_filename", d->eo_narrow_filename,
                     "Model filename for the low gsd eo mode." );
  config->set_value( "eo_medium_filename", d->eo_medium_filename,
                     "Model filename for the medium gsd eo mode." );
  config->set_value( "eo_wide_filename", d->eo_wide_filename,
                     "Model filename for the wide gsd eo mode." );
  config->set_value( "ir_narrow_filename", d->ir_narrow_filename,
                     "Model filename for the low gsd ir mode." );
  config->set_value( "ir_medium_filename", d->ir_medium_filename,
                     "Model filename for the medium gsd ir mode." );
  config->set_value( "ir_wide_filename", d->ir_wide_filename,
                     "Model filename for the wide gsd ir mode." );

  return config;
}

// ----------------------------------------------------------------------------
void
hashed_image_classifier_filter
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed
  // values are present. An alternative is to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->use_variable_models =
    config->get_value< bool >( "use_variable_models" );
  d->lower_gsd_threshold =
    config->get_value< float >( "lower_gsd_threshold" );
  d->upper_gsd_threshold =
    config->get_value< float >( "upper_gsd_threshold" );
  d->default_filename =
    config->get_value< std::string >( "default_filename" );
  d->eo_narrow_filename =
    config->get_value< std::string >( "eo_narrow_filename" );
  d->eo_medium_filename =
    config->get_value< std::string >( "eo_medium_filename" );
  d->eo_wide_filename =
    config->get_value< std::string >( "eo_wide_filename" );
  d->ir_narrow_filename =
    config->get_value< std::string >( "ir_narrow_filename" );
  d->ir_medium_filename =
    config->get_value< std::string >( "ir_medium_filename" );
  d->ir_wide_filename =
    config->get_value< std::string >( "ir_narrow_filename" );
}

// ----------------------------------------------------------------------------
bool
hashed_image_classifier_filter
::check_configuration( vital::config_block_sptr config ) const
{
  float lower_gsd_threshold =
    config->get_value< float >( "lower_gsd_threshold" );
  float upper_gsd_threshold =
    config->get_value< float >( "upper_gsd_threshold" );
  if( lower_gsd_threshold >= upper_gsd_threshold )
  {
    LOG_ERROR( logger(),
               "Lower GSD threshold higher that upper GSD threshold" );
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
hashed_image_classifier_filter
::filter( kwiver::vital::image_container_sptr image_data )
{
  // In the future this shouldd select which model to use
  // based on IO vs. IR and GSD. But for now I'm just going to assume
  // that there's one model which we load at the begining
  // Perform Basic Validation
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );
  vil_image_view< double > weight_image;

  #define HANDLE_CASE( T )                                              \
  case T:                                                               \
  {                                                                     \
    typedef vil_pixel_format_type_of< T >::component_type i_pix;        \
    d->classifier.classify_images( vil_image_view<double>(), weight_image, 0.0 );           \
    break;                                                              \
  }                                                                     \

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );

    default:
    {
      LOG_ERROR( logger(), "Invalid input format type received" );
      return kwiver::vital::image_container_sptr();
    }
  }

  return std::make_shared< vxl::image_container>( weight_image );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
