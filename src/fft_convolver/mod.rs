pub mod fft_convolver;
#[cfg(feature = "ipp")]
pub mod ipp_fft;
pub mod rust_fft;
pub mod traits;
pub mod two_stage_convolver;

use fft_convolver::GenericFFTConvolver;
use traits::{ComplexOps, FftBackend};

#[cfg(not(feature = "ipp"))]
pub use rust_fft::{FFTConvolver, Fft, TwoStageFFTConvolver};

#[cfg(feature = "ipp")]
pub use ipp_fft::{FFTConvolver, Fft, TwoStageFFTConvolver};

#[cfg(all(test, feature = "ipp"))]
mod tests {
    use super::*;
    use crate::{fft_convolver::ipp_fft, fft_convolver::rust_fft, Convolution};

    // Helper function that runs both implementations and compares results
    fn compare_implementations(impulse_response: &[f32], input: &[f32], block_size: usize) {
        let max_len = impulse_response.len();

        // Create both implementations
        let mut rust_convolver =
            rust_fft::FFTConvolver::init(impulse_response, block_size, max_len);
        let mut ipp_convolver = ipp_fft::FFTConvolver::init(impulse_response, block_size, max_len);

        // Prepare output buffers
        let mut rust_output = vec![0.0; input.len()];
        let mut ipp_output = vec![0.0; input.len()];

        // Process with both implementations
        rust_convolver.process(input, &mut rust_output);
        ipp_convolver.process(input, &mut ipp_output);

        // Compare results (accounting for floating-point precision differences)
        for i in 0..input.len() {
            assert!(
                (rust_output[i] - ipp_output[i]).abs() < 1e-5,
                "Outputs differ at position {}: rust={}, ipp={}",
                i,
                rust_output[i],
                ipp_output[i]
            );
        }
    }

    #[test]
    fn test_ipp_vs_rust_impulse() {
        // Test with an impulse response
        let mut response = vec![0.0; 1024];
        response[0] = 1.0;
        let input = vec![1.0; 1024];

        compare_implementations(&response, &input, 256);
    }

    #[test]
    fn test_ipp_vs_rust_decay() {
        // Test with a decaying impulse response
        let mut response = vec![0.0; 1024];
        for i in 0..response.len() {
            response[i] = 0.9f32.powi(i as i32);
        }
        let input = vec![1.0; 1024];

        compare_implementations(&response, &input, 256);
    }

    #[test]
    fn test_ipp_vs_rust_sine() {
        // Test with a sine wave input
        let mut response = vec![0.0; 1024];
        response[0] = 1.0;

        let mut input = vec![0.0; 1024];
        for i in 0..input.len() {
            input[i] = (i as f32 * 0.1).sin();
        }

        compare_implementations(&response, &input, 128);
    }
}
