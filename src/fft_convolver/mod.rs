use crate::Convolution;
#[cfg(feature = "ipp")]
pub mod ipp_fft;
pub mod rust_fft;

// Define a trait that will be implemented by both FFT backends
pub trait FftBackend: Clone {
    type Complex;

    fn init(&mut self, size: usize);
    fn forward(
        &mut self,
        input: &mut [f32],
        output: &mut [Self::Complex],
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn inverse(
        &mut self,
        input: &mut [Self::Complex],
        output: &mut [f32],
    ) -> Result<(), Box<dyn std::error::Error>>;
}

// Define trait for complex operations that will be used by both backends
pub trait ComplexOps: Clone {
    type Complex: Clone + Default;

    fn complex_size(size: usize) -> usize;
    fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize);
    fn complex_multiply_accumulate(
        result: &mut [Self::Complex],
        a: &[Self::Complex],
        b: &[Self::Complex],
        temp_buffer: Option<&mut [Self::Complex]>,
    );
    fn sum(result: &mut [f32], a: &[f32], b: &[f32]);
    fn zero_complex(buffer: &mut [Self::Complex]);
    fn zero_real(buffer: &mut [f32]);
    fn copy_complex(dst: &mut [Self::Complex], src: &[Self::Complex]);
    fn add_to_buffer(dst: &mut [f32], src: &[f32]);
}

// Generic FFTConvolver implementation that works with any backend
#[derive(Default, Clone)]
pub struct GenericFFTConvolver<F: FftBackend, C: ComplexOps<Complex = F::Complex>>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    ir_len: usize,
    block_size: usize,
    _seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    _fft_complex_size: usize,
    segments: Vec<Vec<F::Complex>>,
    segments_ir: Vec<Vec<F::Complex>>,
    fft_buffer: Vec<f32>,
    fft: F,
    pre_multiplied: Vec<F::Complex>,
    conv: Vec<F::Complex>,
    overlap: Vec<f32>,
    current: usize,
    input_buffer: Vec<f32>,
    input_buffer_fill: usize,
    _complex_ops: std::marker::PhantomData<C>,
    temp_buffer: Option<Vec<F::Complex>>,
}

impl<F: FftBackend, C: ComplexOps<Complex = F::Complex>> GenericFFTConvolver<F, C>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    fn init_internal(
        impulse_response: &[f32],
        block_size: usize,
        max_response_length: usize,
    ) -> Self {
        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);
        let ir_len = padded_ir.len();

        let block_size = block_size.next_power_of_two();
        let seg_size = 2 * block_size;
        let seg_count = (ir_len as f64 / block_size as f64).ceil() as usize;
        let active_seg_count = seg_count;
        let fft_complex_size = C::complex_size(seg_size);

        // FFT
        let mut fft = F::default();
        fft.init(seg_size);
        let fft_buffer = vec![0.; seg_size];

        // prepare segments
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            segments.push(vec![F::Complex::default(); fft_complex_size]);
        }

        let mut segments_ir = Vec::with_capacity(seg_count);

        // prepare ir
        let mut ir_buffer = vec![0.0; seg_size];
        for i in 0..seg_count {
            let mut segment = vec![F::Complex::default(); fft_complex_size];
            let remaining = ir_len - (i * block_size);
            let size_copy = if remaining >= block_size {
                block_size
            } else {
                remaining
            };
            C::copy_and_pad(&mut ir_buffer, &padded_ir[i * block_size..], size_copy);
            fft.forward(&mut ir_buffer, &mut segment).unwrap();
            segments_ir.push(segment);
        }

        // prepare convolution buffers
        let pre_multiplied = vec![F::Complex::default(); fft_complex_size];
        let conv = vec![F::Complex::default(); fft_complex_size];
        let overlap = vec![0.; block_size];

        // prepare input buffer
        let input_buffer = vec![0.; block_size];
        let input_buffer_fill = 0;

        // reset current position
        let current = 0;

        // For IPP backend we need a temp buffer
        let temp_buffer = Some(vec![F::Complex::default(); fft_complex_size]);

        Self {
            ir_len,
            block_size,
            _seg_size: seg_size,
            seg_count,
            active_seg_count,
            _fft_complex_size: fft_complex_size,
            segments,
            segments_ir,
            fft_buffer,
            fft,
            pre_multiplied,
            conv,
            overlap,
            current,
            input_buffer,
            input_buffer_fill,
            _complex_ops: std::marker::PhantomData,
            temp_buffer,
        }
    }
}

impl<F: FftBackend, C: ComplexOps<Complex = F::Complex>> Convolution for GenericFFTConvolver<F, C>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    fn init(impulse_response: &[f32], block_size: usize, max_response_length: usize) -> Self {
        Self::init_internal(impulse_response, block_size, max_response_length)
    }

    fn update(&mut self, response: &[f32]) {
        let new_ir_len = response.len();

        if new_ir_len > self.ir_len {
            panic!("New impulse response is longer than initialized length");
        }

        if self.ir_len == 0 {
            return;
        }

        C::zero_real(&mut self.fft_buffer);
        C::zero_complex(&mut self.conv);
        C::zero_complex(&mut self.pre_multiplied);
        C::zero_real(&mut self.overlap);

        self.active_seg_count = ((new_ir_len as f64 / self.block_size as f64).ceil()) as usize;

        // Prepare IR
        for i in 0..self.active_seg_count {
            let segment = &mut self.segments_ir[i];
            let remaining = new_ir_len - (i * self.block_size);
            let size_copy = if remaining >= self.block_size {
                self.block_size
            } else {
                remaining
            };
            C::copy_and_pad(
                &mut self.fft_buffer,
                &response[i * self.block_size..],
                size_copy,
            );
            self.fft.forward(&mut self.fft_buffer, segment).unwrap();
        }

        // Clear remaining segments
        for i in self.active_seg_count..self.seg_count {
            C::zero_complex(&mut self.segments_ir[i]);
        }
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if self.active_seg_count == 0 {
            C::zero_real(output);
            return;
        }

        let mut processed = 0;
        while processed < output.len() {
            let input_buffer_was_empty = self.input_buffer_fill == 0;
            let processing = std::cmp::min(
                output.len() - processed,
                self.block_size - self.input_buffer_fill,
            );

            // Copy input to input buffer
            let input_buffer_pos = self.input_buffer_fill;
            C::copy_and_pad(
                &mut self.input_buffer[input_buffer_pos..],
                &input[processed..processed + processing],
                processing,
            );

            // Forward FFT
            C::copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
            if let Err(_) = self
                .fft
                .forward(&mut self.fft_buffer, &mut self.segments[self.current])
            {
                C::zero_real(output);
                return;
            }

            // complex multiplication
            if input_buffer_was_empty {
                C::zero_complex(&mut self.pre_multiplied);

                for i in 1..self.active_seg_count {
                    let index_ir = i;
                    let index_audio = (self.current + i) % self.active_seg_count;
                    C::complex_multiply_accumulate(
                        &mut self.pre_multiplied,
                        &self.segments_ir[index_ir],
                        &self.segments[index_audio],
                        self.temp_buffer.as_mut().map(|v| v.as_mut_slice()),
                    );
                }
            }

            C::copy_complex(&mut self.conv, &self.pre_multiplied);

            C::complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
                self.temp_buffer.as_mut().map(|v| v.as_mut_slice()),
            );

            // Backward FFT
            if let Err(_) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                C::zero_real(output);
                return;
            }

            // Add overlap
            C::sum(
                &mut output[processed..processed + processing],
                &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                &self.overlap[input_buffer_pos..input_buffer_pos + processing],
            );

            // Input buffer full => Next block
            self.input_buffer_fill += processing;
            if self.input_buffer_fill == self.block_size {
                // Input buffer is empty again now
                C::zero_real(&mut self.input_buffer);
                self.input_buffer_fill = 0;

                // Save the overlap
                C::copy_and_pad(
                    &mut self.overlap,
                    &self.fft_buffer[self.block_size..],
                    self.block_size,
                );

                // Update the current segment
                self.current = if self.current > 0 {
                    self.current - 1
                } else {
                    self.active_seg_count - 1
                };
            }
            processed += processing;
        }
    }
}

// Generic TwoStageFFTConvolver implementation
#[derive(Clone)]
pub struct GenericTwoStageFFTConvolver<F: FftBackend, C: ComplexOps<Complex = F::Complex>>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    head_convolver: GenericFFTConvolver<F, C>,
    tail_convolver0: GenericFFTConvolver<F, C>,
    tail_output0: Vec<f32>,
    tail_precalculated0: Vec<f32>,
    tail_convolver: GenericFFTConvolver<F, C>,
    tail_output: Vec<f32>,
    tail_precalculated: Vec<f32>,
    tail_input: Vec<f32>,
    tail_input_fill: usize,
    precalculated_pos: usize,
    head_block_size: usize,
    tail_block_size: usize,
}

impl<F: FftBackend, C: ComplexOps<Complex = F::Complex>> GenericTwoStageFFTConvolver<F, C>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    pub fn init(
        impulse_response: &[f32],
        _block_size: usize,
        max_response_length: usize,
        head_block_size: usize,
        tail_block_size: usize,
    ) -> Self {
        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);

        let head_ir_len = std::cmp::min(max_response_length, tail_block_size);
        let head_convolver = GenericFFTConvolver::<F, C>::init(
            &padded_ir[0..head_ir_len],
            head_block_size,
            max_response_length,
        );

        let tail_convolver0 = if max_response_length > tail_block_size {
            let tail_ir_len = std::cmp::min(max_response_length - tail_block_size, tail_block_size);
            GenericFFTConvolver::<F, C>::init(
                &padded_ir[tail_block_size..tail_block_size + tail_ir_len],
                head_block_size,
                max_response_length,
            )
        } else {
            GenericFFTConvolver::<F, C>::default()
        };

        let tail_output0 = vec![0.0; tail_block_size];
        let tail_precalculated0 = vec![0.0; tail_block_size];

        let tail_convolver = if max_response_length > 2 * tail_block_size {
            let tail_ir_len = max_response_length - 2 * tail_block_size;
            GenericFFTConvolver::<F, C>::init(
                &padded_ir[2 * tail_block_size..2 * tail_block_size + tail_ir_len],
                tail_block_size,
                max_response_length,
            )
        } else {
            GenericFFTConvolver::<F, C>::default()
        };

        let tail_output = vec![0.0; tail_block_size];
        let tail_precalculated = vec![0.0; tail_block_size];
        let tail_input = vec![0.0; tail_block_size];
        let tail_input_fill = 0;
        let precalculated_pos = 0;

        GenericTwoStageFFTConvolver {
            head_convolver,
            tail_convolver0,
            tail_output0,
            tail_precalculated0,
            tail_convolver,
            tail_output,
            tail_precalculated,
            tail_input,
            tail_input_fill,
            precalculated_pos,
            head_block_size,
            tail_block_size,
        }
    }

    pub fn update(&mut self, _response: &[f32]) {
        todo!()
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let head_block_size = self.head_block_size;
        let tail_block_size = self.tail_block_size;

        // Head
        self.head_convolver.process(input, output);

        // Tail
        if self.tail_input.is_empty() {
            return;
        }

        let len = input.len();
        let mut processed = 0;

        while processed < len {
            let remaining = len - processed;
            let processing = std::cmp::min(
                remaining,
                head_block_size - (self.tail_input_fill % head_block_size),
            );

            // Sum head and tail
            let sum_begin = processed;

            // Sum: 1st tail block
            if !self.tail_precalculated0.is_empty() {
                C::add_to_buffer(
                    &mut output[sum_begin..sum_begin + processing],
                    &self.tail_precalculated0
                        [self.precalculated_pos..self.precalculated_pos + processing],
                );
            }

            // Sum: 2nd-Nth tail block
            if !self.tail_precalculated.is_empty() {
                C::add_to_buffer(
                    &mut output[sum_begin..sum_begin + processing],
                    &self.tail_precalculated
                        [self.precalculated_pos..self.precalculated_pos + processing],
                );
            }
            self.precalculated_pos += processing;

            // Fill input buffer for tail convolution
            C::copy_and_pad(
                &mut self.tail_input[self.tail_input_fill..],
                &input[processed..processed + processing],
                processing,
            );
            self.tail_input_fill += processing;

            // Convolution: 1st tail block
            if !self.tail_precalculated0.is_empty() && self.tail_input_fill % head_block_size == 0 {
                assert!(self.tail_input_fill >= head_block_size);
                let block_offset = self.tail_input_fill - head_block_size;
                self.tail_convolver0.process(
                    &self.tail_input[block_offset..block_offset + head_block_size],
                    &mut self.tail_output0[block_offset..block_offset + head_block_size],
                );
                if self.tail_input_fill == tail_block_size {
                    std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                }
            }

            // Convolution: 2nd-Nth tail block (might be done in some background thread)
            if !self.tail_precalculated.is_empty()
                && self.tail_input_fill == tail_block_size
                && self.tail_output.len() == tail_block_size
            {
                std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                self.tail_convolver
                    .process(&self.tail_input, &mut self.tail_output);
            }

            if self.tail_input_fill == tail_block_size {
                self.tail_input_fill = 0;
                self.precalculated_pos = 0;
            }

            processed += processing;
        }
    }
}

// Type definitions for IPP and RustFFT implementations
#[cfg(not(feature = "ipp"))]
pub use rust_fft::{
    complex_multiply_accumulate, complex_size, copy_and_pad, sum, FFTConvolver, Fft,
    TwoStageFFTConvolver,
};
pub mod rust_ops {
    use super::*;
    use rustfft::num_complex::Complex;

    #[derive(Clone, Default)]
    pub struct RustComplexOps;

    impl ComplexOps for RustComplexOps {
        type Complex = Complex<f32>;

        fn complex_size(size: usize) -> usize {
            rust_fft::complex_size(size)
        }

        fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
            rust_fft::copy_and_pad(dst, src, src_size)
        }

        fn complex_multiply_accumulate(
            result: &mut [Self::Complex],
            a: &[Self::Complex],
            b: &[Self::Complex],
            _temp_buffer: Option<&mut [Self::Complex]>,
        ) {
            rust_fft::complex_multiply_accumulate(result, a, b)
        }

        fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
            rust_fft::sum(result, a, b)
        }

        fn zero_complex(buffer: &mut [Self::Complex]) {
            buffer.fill(Complex { re: 0.0, im: 0.0 });
        }

        fn zero_real(buffer: &mut [f32]) {
            buffer.fill(0.0);
        }

        fn copy_complex(dst: &mut [Self::Complex], src: &[Self::Complex]) {
            dst.clone_from_slice(src);
        }

        fn add_to_buffer(dst: &mut [f32], src: &[f32]) {
            rust_fft::add_to_buffer(dst, src);
        }
    }
}

#[cfg(feature = "ipp")]
pub use ipp_fft::{
    complex_multiply_accumulate, complex_size, copy_and_pad, sum, FFTConvolver, Fft,
    TwoStageFFTConvolver,
};
#[cfg(feature = "ipp")]
mod ipp_ops {
    use super::*;
    use ipp_sys::*;
    use num_complex::Complex32;

    #[derive(Clone, Default)]
    pub struct IppComplexOps;

    impl ComplexOps for IppComplexOps {
        type Complex = Complex32;

        fn complex_size(size: usize) -> usize {
            ipp_fft::complex_size(size)
        }

        fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
            ipp_fft::copy_and_pad(dst, src, src_size)
        }

        fn complex_multiply_accumulate(
            result: &mut [Self::Complex],
            a: &[Self::Complex],
            b: &[Self::Complex],
            temp_buffer: Option<&mut [Self::Complex]>,
        ) {
            let temp = temp_buffer.expect("IPP implementation requires a temp buffer");
            ipp_fft::complex_multiply_accumulate(result, a, b, temp)
        }

        fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
            ipp_fft::sum(result, a, b)
        }

        fn zero_complex(buffer: &mut [Self::Complex]) {
            unsafe {
                ippsZero_32fc(
                    buffer.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                    buffer.len() as i32,
                );
            }
        }

        fn zero_real(buffer: &mut [f32]) {
            unsafe {
                ippsZero_32f(buffer.as_mut_ptr(), buffer.len() as i32);
            }
        }

        fn copy_complex(dst: &mut [Self::Complex], src: &[Self::Complex]) {
            unsafe {
                ippsCopy_32fc(
                    src.as_ptr() as *const ipp_sys::Ipp32fc,
                    dst.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                    dst.len() as i32,
                );
            }
        }

        fn add_to_buffer(dst: &mut [f32], src: &[f32]) {
            unsafe {
                ippsAdd_32f_I(src.as_ptr(), dst.as_mut_ptr(), dst.len() as i32);
            }
        }
    }
}

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
