#[cfg(test)]
mod tests {
    use crate::crossfade_convolver::CrossfadeConvolver;
    use crate::fft_convolver::FFTConvolver;
    use crate::{Convolution, Sample};

    #[test]
    fn test_fft_convolver_passthrough() {
        let mut response = [0.0; 1024];
        response[0] = 1.0;
        let mut convolver = FFTConvolver::init(&response, 1024, response.len());
        let input = vec![1.0; 1024];
        let mut output = vec![0.0; 1024];
        convolver.process(&input, &mut output);

        for i in 0..1024 {
            assert!((output[i] - 1.0).abs() < 1e-6);
        }
    }

    fn generate_sinusoid(
        length: usize,
        frequency: f32,
        sample_rate: f32,
        gain: f32,
    ) -> Vec<Sample> {
        let mut signal = vec![0.0; length];
        for i in 0..length {
            signal[i] =
                gain * (2.0 * std::f32::consts::PI * frequency * i as Sample / sample_rate).sin();
        }
        signal
    }

    #[test]
    fn fft_convolver_update_is_reset() {
        let block_size = 512;
        let response_a = generate_sinusoid(block_size, 1000.0, 48000.0, 1.0);
        let response_b = generate_sinusoid(block_size, 2000.0, 48000.0, 0.7);
        let mut convolver_a = FFTConvolver::init(&response_a, block_size, response_a.len());
        let mut convolver_b = FFTConvolver::init(&response_b, block_size, response_b.len());
        let mut convolver_update = FFTConvolver::init(&response_a, block_size, response_a.len());
        let mut output_a = vec![0.0; block_size];
        let mut output_b = vec![0.0; block_size];
        let mut output_update = vec![0.0; block_size];

        let num_input_blocks = 16;
        let input = generate_sinusoid(num_input_blocks * block_size, 1300.0, 48000.0, 1.0);

        let update_index = 8;

        for i in 0..num_input_blocks {
            if i == update_index {
                convolver_update.update(&response_b);
            }

            convolver_update.process(
                &input[i * block_size..(i + 1) * block_size],
                &mut output_update,
            );

            let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
                for j in 0..block_size {
                    assert!((lhs[j] - rhs[j]).abs() < 1e-6);
                }
            };

            if i < update_index {
                convolver_a.process(&input[i * block_size..(i + 1) * block_size], &mut output_a);
                check_equal(&output_a, &output_update);
            } else {
                convolver_b.process(&input[i * block_size..(i + 1) * block_size], &mut output_b);
                check_equal(&output_b, &output_update);
            }
        }
    }

    #[test]
    fn test_crossfade_convolver() {
        let block_size = 512;
        let response_a = generate_sinusoid(block_size, 1000.0, 48000.0, 1.0);
        let response_b = generate_sinusoid(block_size, 2000.0, 48000.0, 0.7);
        let mut convolver_a = FFTConvolver::init(&response_a, block_size, response_a.len());
        let mut convolver_b = FFTConvolver::init(&response_b, block_size, response_b.len());
        let mut crossfade_convolver =
            CrossfadeConvolver::new(convolver_a.clone(), block_size, block_size, block_size);
        let mut output_a = vec![0.0; block_size];
        let mut output_b = vec![0.0; block_size];
        let mut output_crossfade_convolver = vec![0.0; block_size];

        let num_input_blocks = 16;
        let input = generate_sinusoid(num_input_blocks * block_size, 1300.0, 48000.0, 1.0);

        let update_index = 8;

        for i in 0..num_input_blocks {
            if i == update_index {
                crossfade_convolver.update(&response_b);
            }

            crossfade_convolver.process(
                &input[i * block_size..(i + 1) * block_size],
                &mut output_crossfade_convolver,
            );

            let check_equal = |lhs: &[Sample], rhs: &[Sample]| {
                for j in 0..block_size {
                    assert!((lhs[j] - rhs[j]).abs() < 1e-6);
                }
            };

            convolver_a.process(&input[i * block_size..(i + 1) * block_size], &mut output_a);
            if i >= update_index {
                convolver_b.process(&input[i * block_size..(i + 1) * block_size], &mut output_b);
            }

            if i <= update_index {
                check_equal(&output_a, &output_crossfade_convolver);
            } else {
                if i == update_index + 1 {
                    // crossover sample
                    let crossover_index = block_size / 2 - 1;
                    assert!(
                        (output_crossfade_convolver[crossover_index]
                            - (output_a[crossover_index] * 0.5 + output_b[crossover_index] * 0.5))
                            .abs()
                            < 1e-6
                    );
                } else {
                    check_equal(&output_b, &output_crossfade_convolver);
                }
            }
        }
    }

    #[cfg(feature = "ipp")]
    mod ipp_tests {
        use crate::fft_convolver::{ipp_fft, rust_fft};
        use crate::Convolution;

        #[test]
        fn test_fft_convolver_passthrough() {
            let mut response = [0.0; 1024];
            response[0] = 1.0;
            let mut convolver = ipp_fft::FFTConvolver::init(&response, 1024, response.len());
            let input = vec![1.0; 1024];
            let mut output = vec![0.0; 1024];
            convolver.process(&input, &mut output);

            for i in 0..1024 {
                assert!((output[i] - 1.0).abs() < 1e-6);
            }
        }
        // Helper function that runs both implementations and compares results
        fn compare_implementations(impulse_response: &[f32], input: &[f32], block_size: usize) {
            let max_len = impulse_response.len();

            let mut rust_convolver =
                rust_fft::FFTConvolver::init(impulse_response, block_size, max_len);
            let mut ipp_convolver =
                ipp_fft::FFTConvolver::init(impulse_response, block_size, max_len);

            let mut rust_output = vec![0.0; input.len()];
            let mut ipp_output = vec![0.0; input.len()];

            rust_convolver.process(input, &mut rust_output);
            ipp_convolver.process(input, &mut ipp_output);

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
            let mut response = vec![0.0; 1024];
            response[0] = 1.0;
            let input = vec![1.0; 1024];

            compare_implementations(&response, &input, 256);
        }

        #[test]
        fn test_ipp_vs_rust_decay() {
            let mut response = vec![0.0; 1024];
            for i in 0..response.len() {
                response[i] = 0.9f32.powi(i as i32);
            }
            let input = vec![1.0; 1024];

            compare_implementations(&response, &input, 256);
        }

        #[test]
        fn test_ipp_vs_rust_sine() {
            let mut response = vec![0.0; 1024];
            response[0] = 1.0;

            let mut input = vec![0.0; 1024];
            for i in 0..input.len() {
                input[i] = (i as f32 * 0.1).sin();
            }

            compare_implementations(&response, &input, 128);
        }
    }
}
