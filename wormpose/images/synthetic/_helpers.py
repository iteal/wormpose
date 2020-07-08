import cv2
import numpy as np

from wormpose.images.worm_drawing import make_draw_worm_body
from wormpose.pose.centerline import calculate_skeleton


class TargetSkeletonCalculator(object):
    """
    Calculates the target skeleton
    Makes sure the target skeleton has the same dimensions as the template skeleton by interpolation if necessary
    """

    def __init__(self, nb_skeletons_joints, theta_dims, output_image_shape):
        self.output_image_shape = output_image_shape
        self.need_interpolation = nb_skeletons_joints != theta_dims
        if self.need_interpolation:
            self.interp_x = np.linspace(0, theta_dims - 1, nb_skeletons_joints)
            self.interp_xp = np.arange(0, theta_dims)
        self.non_interpolated_skeleton = np.empty((theta_dims, 2), dtype=float)

    def calculate(self, theta, worm_length, out):

        calculate_skeleton(
            theta, worm_length, canvas_width_height=self.output_image_shape, out=self.non_interpolated_skeleton,
        )

        if self.need_interpolation:
            self._interpolate(self.non_interpolated_skeleton[:, 0], dst=out[:, 0])
            self._interpolate(self.non_interpolated_skeleton[:, 1], dst=out[:, 1])
        else:
            np.copyto(dst=out, src=self.non_interpolated_skeleton)

    def _interpolate(self, non_interpolated, dst):
        np.copyto(dst=dst, src=np.interp(self.interp_x, self.interp_xp, non_interpolated))


class WormOutlineMask(object):
    """
    Filters an image by a mask representing the global outline of the worm,
    to smooth out the rough edges of assembling the body segment image patches
    """

    def __init__(self, output_image_shape):
        self.draw_worm_body = make_draw_worm_body()
        self.target_shape_outline_image = np.empty(output_image_shape, dtype=np.uint8)

    def apply(self, worm_thickness, output_image, target_skel):
        # reset mask
        self.target_shape_outline_image.fill(0)

        # draw the target worm outline on the mask
        self.draw_worm_body(worm_thickness, self.target_shape_outline_image, skeleton=target_skel)

        # filter the synthetic image with the mask of the worm outline
        output_image[self.target_shape_outline_image == 0] = 0


class PatchDrawing(object):
    """
    Transform image patches of body segments between a template image to a target image
    """

    def __init__(self, output_image_shape):
        self.source_image_mask = None
        self.output_image_shape = output_image_shape

        # preallocate arrays for image processing operations
        self.target_image_mask = np.empty(self.output_image_shape, dtype=np.uint8)
        self.cur_alpha = np.empty(self.output_image_shape, dtype=np.float32)
        self.all_alpha = np.empty(self.output_image_shape, dtype=np.float32)
        self.target_image = np.empty(self.output_image_shape, dtype=np.uint8)

    def draw_patch(self, template_coords, target_coords, recentered_frame, out_image):
        # grow preallocated image as necessary
        if (
            self.source_image_mask is None
            or self.source_image_mask.shape[0] < recentered_frame.shape[0]
            or self.source_image_mask.shape[1] < recentered_frame.shape[1]
        ):
            self.source_image_mask = np.empty_like(recentered_frame)
        source_image_mask = self.source_image_mask[: recentered_frame.shape[0], : recentered_frame.shape[1]]

        transform = cv2.getAffineTransform(target_coords[:3], template_coords[:3])
        source_image_mask.fill(0)
        cv2.fillConvexPoly(source_image_mask, target_coords.astype(np.int), 255)
        masked_frame = cv2.bitwise_and(recentered_frame, source_image_mask)
        cv2.warpAffine(
            masked_frame,
            transform,
            dsize=(self.output_image_shape[1], self.output_image_shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            flags=cv2.INTER_NEAREST,
            dst=self.target_image,
        )
        self.target_image_mask.fill(0)
        cv2.fillConvexPoly(self.target_image_mask, template_coords.astype(np.int), 255)
        cv2.bitwise_and(self.target_image, self.target_image_mask, dst=self.target_image)

        np.greater(self.target_image, 0, out=self.cur_alpha)
        np.greater(out_image, 0, out=self.all_alpha)
        self.all_alpha += self.cur_alpha
        self.all_alpha[self.all_alpha > 1] = 0.5
        out_image += self.target_image
        out_image *= self.all_alpha
