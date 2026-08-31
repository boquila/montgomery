use burn::tensor::{Tensor, backend::Backend, ops::PadMode};

/// Exact zero-padded depthwise 3x3 cross-correlation for training graphs.
///
/// Burn/WGPU currently sends grouped-convolution gradients through its generic fallback. This
/// channelwise stencil keeps the original parameter and gradients while avoiding that path.
pub(crate) fn depthwise_3x3_stride_1<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    debug_assert_eq!(weight.dims(), [channels, 1, 3, 3]);
    let padded = input.pad((1, 1, 1, 1), PadMode::Constant(0.0));
    let term = |kernel_y: usize, kernel_x: usize| {
        padded.clone().slice([
            0..batch,
            0..channels,
            kernel_y..kernel_y + height,
            kernel_x..kernel_x + width,
        ]) * weight
            .clone()
            .slice([
                0..channels,
                0..1,
                kernel_y..kernel_y + 1,
                kernel_x..kernel_x + 1,
            ])
            .reshape([1, channels, 1, 1])
    };
    let mut output = term(0, 0);
    for kernel_y in 0..3 {
        for kernel_x in 0..3 {
            if kernel_y != 0 || kernel_x != 0 {
                output = output + term(kernel_y, kernel_x);
            }
        }
    }
    output
}
