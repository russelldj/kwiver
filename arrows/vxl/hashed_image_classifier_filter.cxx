// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "hashed_image_classifier.h"
#include "hashed_image_classifier_filter.h"

#include <arrows/vxl/image_container.h>

#include <vil/algo/vil_threshold.h>
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

  // --------------------------------------------------------------------------
  priv( hashed_image_classifier_filter* parent ) : p{ parent }
  {
  }

  // Convert the type
  template < typename ipix_t > vil_image_view< ipix_t >
  convert( vil_image_view_base_sptr& view );

  // Scale and convert the image
  bool
  load_model();

  hashed_image_classifier_filter* p;

  vidtk::hashed_image_classifier< vxl_byte, double > hashed_classifier;
  bool model_loaded = false;

  bool use_variable_models = false;
  float lower_gsd_threshold{ 0.11 };
  float upper_gsd_threshold{ 0.22 };

  std::string default_filename = "";
  std::string eo_narrow_filename = "";
  std::string eo_medium_filename = "";
  std::string eo_wide_filename = "";
  std::string ir_narrow_filename = "";
  std::string ir_medium_filename = "";
  std::string ir_wide_filename = "";
};

// ----------------------------------------------------------------------------
bool
hashed_image_classifier_filter::priv
::load_model()
{
  if( !model_loaded )
  {
    if( !hashed_classifier.load_from_file( default_filename ) )
    {
      LOG_ERROR( p->logger(),
                 "Could not load default_filename model" );
      return false;
    }
    model_loaded = true;
  }
  return true;
}

// ----------------------------------------------------------------------------
hashed_image_classifier_filter
::hashed_image_classifier_filter()
  : d( new priv( this ) )
{
  attach_logger( "arrows.vxl.hashed_image_classifier_filter" );
}

// ----------------------------------------------------------------------------
hashed_image_classifier_filter
::~hashed_image_classifier_filter()
{
}

// ----------------------------------------------------------------------------
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
  // Start with our generated vital::config_block to ensure that assumed values
  // are present. An alternative would be to check for key presence before
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
               "Lower GSD threshold higher than upper GSD threshold" );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
hashed_image_classifier_filter
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    LOG_ERROR( logger(), "Image pointer was null" );
    return kwiver::vital::image_container_sptr();
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  if( !view )
  {
    LOG_ERROR( logger(), "Data contained in the image container is null" );
    return nullptr;
  }

  if( view->pixel_format() != VIL_PIXEL_FORMAT_BYTE )
  {
    LOG_ERROR( logger(), "Only byte images can be proccessed" );
    return nullptr;
  }

  if( !d->load_model() )
  {
    return nullptr;
  }

  vil_image_view< double > weight_image;

  // TODO decide if this offset should be tunable
  d->hashed_classifier.classify_images( view, weight_image, 0.0 );

  vil_image_view< vxl_byte > binarized;
  vil_transform( weight_image, binarized, []( double pix ){
                   return pix < 0 ? 0 : 255;
                 } );

  return std::make_shared< vxl::image_container >( binarized );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
