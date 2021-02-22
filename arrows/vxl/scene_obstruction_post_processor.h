// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VXL_SCENE_OBSTRUCTION_POST_PROCESS_
#define KWIVER_ARROWS_VXL_SCENE_OBSTRUCTION_POST_PROCESS_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {

namespace arrows {

namespace vxl {

/// VXL Scene obstruction post processor.
///
/// A stateful post processing approach which takes classification scores over
/// time and produces a mask of the obstructions
class KWIVER_ALGO_VXL_EXPORT scene_obstruction_post_processor
  : public vital::algo::image_filter
{
public:
  PLUGIN_INFO( "vxl_scene_obstruction_post_processor",
               "Use VXL to average frames together." )

  scene_obstruction_post_processor();
  virtual ~scene_obstruction_post_processor();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink.
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block.
  virtual void set_configuration( vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid.
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Produce the updated mask.
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  class priv;

  std::unique_ptr< priv > const d;
};

} // namespace vxl

} // namespace arrows

} // namespace kwiver

#endif
