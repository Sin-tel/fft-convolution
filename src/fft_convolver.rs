mod rust_fft {
    use crate::{Convolution, Sample};
    use realfft::{ComplexToReal, FftError, RealFftPlanner, RealToComplex};
    use rustfft::num_complex::Complex;
    use std::sync::Arc;
    #[derive(Clone)]
    pub struct Fft {
        fft_forward: Arc<dyn RealToComplex<f32>>,
        fft_inverse: Arc<dyn ComplexToReal<f32>>,
    }

    impl Default for Fft {
        fn default() -> Self {
            let mut planner = RealFftPlanner::<f32>::new();
            Self {
                fft_forward: planner.plan_fft_forward(0),
                fft_inverse: planner.plan_fft_inverse(0),
            }
        }
    }

    impl std::fmt::Debug for Fft {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "")
        }
    }

    impl Fft {
        pub fn init(&mut self, length: usize) {
            let mut planner = RealFftPlanner::<f32>::new();
            self.fft_forward = planner.plan_fft_forward(length);
            self.fft_inverse = planner.plan_fft_inverse(length);
        }

        pub fn forward(
            &self,
            input: &mut [f32],
            output: &mut [Complex<f32>],
        ) -> Result<(), FftError> {
            self.fft_forward.process(input, output)?;
            Ok(())
        }

        pub fn inverse(
            &self,
            input: &mut [Complex<f32>],
            output: &mut [f32],
        ) -> Result<(), FftError> {
            self.fft_inverse.process(input, output)?;

            // FFT Normalization
            let len = output.len();
            output.iter_mut().for_each(|bin| *bin /= len as f32);

            Ok(())
        }
    }
    #[derive(Default, Clone)]
    pub struct FFTConvolver {
        ir_len: usize,
        block_size: usize,
        _seg_size: usize,
        seg_count: usize,
        active_seg_count: usize,
        _fft_complex_size: usize,
        segments: Vec<Vec<Complex<f32>>>,
        segments_ir: Vec<Vec<Complex<f32>>>,
        fft_buffer: Vec<f32>,
        fft: Fft,
        pre_multiplied: Vec<Complex<f32>>,
        conv: Vec<Complex<f32>>,
        overlap: Vec<f32>,
        current: usize,
        input_buffer: Vec<f32>,
        input_buffer_fill: usize,
    }

    impl Convolution for FFTConvolver {
        fn init(
            impulse_response: &[Sample],
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
            let fft_complex_size = complex_size(seg_size);

            // FFT
            let mut fft = Fft::default();
            fft.init(seg_size);
            let mut fft_buffer = vec![0.; seg_size];

            // prepare segments
            let segments = vec![vec![Complex::new(0., 0.); fft_complex_size]; seg_count];
            let mut segments_ir = Vec::new();

            // prepare ir
            for i in 0..seg_count {
                let mut segment = vec![Complex::new(0., 0.); fft_complex_size];
                let remaining = ir_len - (i * block_size);
                let size_copy = if remaining >= block_size {
                    block_size
                } else {
                    remaining
                };
                copy_and_pad(&mut fft_buffer, &padded_ir[i * block_size..], size_copy);
                fft.forward(&mut fft_buffer, &mut segment).unwrap();
                segments_ir.push(segment);
            }

            // prepare convolution buffers
            let pre_multiplied = vec![Complex::new(0., 0.); fft_complex_size];
            let conv = vec![Complex::new(0., 0.); fft_complex_size];
            let overlap = vec![0.; block_size];

            // prepare input buffer
            let input_buffer = vec![0.; block_size];
            let input_buffer_fill = 0;

            // reset current position
            let current = 0;

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
            }
        }

        fn update(&mut self, response: &[Sample]) {
            let new_ir_len = response.len();

            if new_ir_len > self.ir_len {
                panic!("New impulse response is longer than initialized length");
            }

            if self.ir_len == 0 {
                return;
            }

            self.fft_buffer.fill(0.);
            self.conv.fill(Complex::new(0., 0.));
            self.pre_multiplied.fill(Complex::new(0., 0.));
            self.overlap.fill(0.);

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
                copy_and_pad(
                    &mut self.fft_buffer,
                    &response[i * self.block_size..],
                    size_copy,
                );
                self.fft.forward(&mut self.fft_buffer, segment).unwrap();
            }

            // Clear remaining segments
            for i in self.active_seg_count..self.seg_count {
                self.segments_ir[i].fill(Complex::new(0., 0.));
            }
        }

        fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
            if self.active_seg_count == 0 {
                output.fill(0.);
                return;
            }

            let mut processed = 0;
            while processed < output.len() {
                let input_buffer_was_empty = self.input_buffer_fill == 0;
                let processing = std::cmp::min(
                    output.len() - processed,
                    self.block_size - self.input_buffer_fill,
                );

                let input_buffer_pos = self.input_buffer_fill;
                self.input_buffer[input_buffer_pos..input_buffer_pos + processing]
                    .clone_from_slice(&input[processed..processed + processing]);

                // Forward FFT
                copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
                if let Err(_err) = self
                    .fft
                    .forward(&mut self.fft_buffer, &mut self.segments[self.current])
                {
                    output.fill(0.);
                    return; // error!
                }

                // complex multiplication
                if input_buffer_was_empty {
                    self.pre_multiplied.fill(Complex { re: 0., im: 0. });
                    for i in 1..self.active_seg_count {
                        let index_ir = i;
                        let index_audio = (self.current + i) % self.active_seg_count;
                        complex_multiply_accumulate(
                            &mut self.pre_multiplied,
                            &self.segments_ir[index_ir],
                            &self.segments[index_audio],
                        );
                    }
                }
                self.conv.clone_from_slice(&self.pre_multiplied);
                complex_multiply_accumulate(
                    &mut self.conv,
                    &self.segments[self.current],
                    &self.segments_ir[0],
                );

                // Backward FFT
                if let Err(_err) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                    output.fill(0.);
                    return; // error!
                }

                // Add overlap
                sum(
                    &mut output[processed..processed + processing],
                    &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                    &self.overlap[input_buffer_pos..input_buffer_pos + processing],
                );

                // Input buffer full => Next block
                self.input_buffer_fill += processing;
                if self.input_buffer_fill == self.block_size {
                    // Input buffer is empty again now
                    self.input_buffer.fill(0.);
                    self.input_buffer_fill = 0;
                    // Save the overlap
                    self.overlap
                        .clone_from_slice(&self.fft_buffer[self.block_size..self.block_size * 2]);

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
    pub fn complex_size(size: usize) -> usize {
        (size / 2) + 1
    }

    pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
        assert!(dst.len() >= src_size);
        dst[0..src_size].clone_from_slice(&src[0..src_size]);
        dst[src_size..].iter_mut().for_each(|value| *value = 0.);
    }

    pub fn complex_multiply_accumulate(
        result: &mut [Complex<f32>],
        a: &[Complex<f32>],
        b: &[Complex<f32>],
    ) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());
        let len = result.len();
        let end4 = 4 * (len / 4);
        for i in (0..end4).step_by(4) {
            result[i + 0].re += a[i + 0].re * b[i + 0].re - a[i + 0].im * b[i + 0].im;
            result[i + 1].re += a[i + 1].re * b[i + 1].re - a[i + 1].im * b[i + 1].im;
            result[i + 2].re += a[i + 2].re * b[i + 2].re - a[i + 2].im * b[i + 2].im;
            result[i + 3].re += a[i + 3].re * b[i + 3].re - a[i + 3].im * b[i + 3].im;
            result[i + 0].im += a[i + 0].re * b[i + 0].im + a[i + 0].im * b[i + 0].re;
            result[i + 1].im += a[i + 1].re * b[i + 1].im + a[i + 1].im * b[i + 1].re;
            result[i + 2].im += a[i + 2].re * b[i + 2].im + a[i + 2].im * b[i + 2].re;
            result[i + 3].im += a[i + 3].re * b[i + 3].im + a[i + 3].im * b[i + 3].re;
        }
        for i in end4..len {
            result[i].re += a[i].re * b[i].re - a[i].im * b[i].im;
            result[i].im += a[i].re * b[i].im + a[i].im * b[i].re;
        }
    }

    pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());
        let len = result.len();
        let end4 = 3 * (len / 4);
        for i in (0..end4).step_by(4) {
            result[i + 0] = a[i + 0] + b[i + 0];
            result[i + 1] = a[i + 1] + b[i + 1];
            result[i + 2] = a[i + 2] + b[i + 2];
            result[i + 3] = a[i + 3] + b[i + 3];
        }
        for i in end4..len {
            result[i] = a[i] + b[i];
        }
    }
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
    #[derive(Clone)]
    pub struct TwoStageFFTConvolver {
        head_convolver: FFTConvolver,
        tail_convolver0: FFTConvolver,
        tail_output0: Vec<Sample>,
        tail_precalculated0: Vec<Sample>,
        tail_convolver: FFTConvolver,
        tail_output: Vec<Sample>,
        tail_precalculated: Vec<Sample>,
        tail_input: Vec<Sample>,
        tail_input_fill: usize,
        precalculated_pos: usize,
    }

    const HEAD_BLOCK_SIZE: usize = 128;
    const TAIL_BLOCK_SIZE: usize = 1024;

    impl Convolution for TwoStageFFTConvolver {
        fn init(
            impulse_response: &[Sample],
            _block_size: usize,
            max_response_length: usize,
        ) -> Self {
            let head_block_size = HEAD_BLOCK_SIZE;
            let tail_block_size = TAIL_BLOCK_SIZE;

            if max_response_length < impulse_response.len() {
                panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
            }
            let mut padded_ir = impulse_response.to_vec();
            padded_ir.resize(max_response_length, 0.);

            let head_ir_len = std::cmp::min(max_response_length, tail_block_size);
            let head_convolver = FFTConvolver::init(
                &padded_ir[0..head_ir_len],
                head_block_size,
                max_response_length,
            );

            let tail_convolver0 = (max_response_length > tail_block_size)
                .then(|| {
                    let tail_ir_len =
                        std::cmp::min(max_response_length - tail_block_size, tail_block_size);
                    FFTConvolver::init(
                        &padded_ir[tail_block_size..tail_block_size + tail_ir_len],
                        head_block_size,
                        max_response_length,
                    )
                })
                .unwrap_or_default();

            let tail_output0 = vec![0.0; tail_block_size];
            let tail_precalculated0 = vec![0.0; tail_block_size];

            let tail_convolver = (max_response_length > 2 * tail_block_size)
                .then(|| {
                    let tail_ir_len = max_response_length - 2 * tail_block_size;
                    FFTConvolver::init(
                        &padded_ir[2 * tail_block_size..2 * tail_block_size + tail_ir_len],
                        tail_block_size,
                        max_response_length,
                    )
                })
                .unwrap_or_default();

            let tail_output = vec![0.0; tail_block_size];
            let tail_precalculated = vec![0.0; tail_block_size];
            let tail_input = vec![0.0; tail_block_size];
            let tail_input_fill = 0;
            let precalculated_pos = 0;

            TwoStageFFTConvolver {
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
            }
        }

        fn update(&mut self, _response: &[Sample]) {
            todo!()
        }

        fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
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
                    HEAD_BLOCK_SIZE - (self.tail_input_fill % HEAD_BLOCK_SIZE),
                );

                // Sum head and tail
                let sum_begin = processed;
                let sum_end = processed + processing;

                // Sum: 1st tail block
                if self.tail_precalculated0.len() > 0 {
                    let mut precalculated_pos = self.precalculated_pos;
                    for i in sum_begin..sum_end {
                        output[i] += self.tail_precalculated0[precalculated_pos];
                        precalculated_pos += 1;
                    }
                }

                // Sum: 2nd-Nth tail block
                if self.tail_precalculated.len() > 0 {
                    let mut precalculated_pos = self.precalculated_pos;
                    for i in sum_begin..sum_end {
                        output[i] += self.tail_precalculated[precalculated_pos];
                        precalculated_pos += 1;
                    }
                }

                self.precalculated_pos += processing;

                // Fill input buffer for tail convolution
                self.tail_input[self.tail_input_fill..self.tail_input_fill + processing]
                    .copy_from_slice(&input[processed..processed + processing]);
                self.tail_input_fill += processing;

                // Convolution: 1st tail block
                if self.tail_precalculated0.len() > 0 && self.tail_input_fill % HEAD_BLOCK_SIZE == 0
                {
                    assert!(self.tail_input_fill >= HEAD_BLOCK_SIZE);
                    let block_offset = self.tail_input_fill - HEAD_BLOCK_SIZE;
                    self.tail_convolver0.process(
                        &self.tail_input[block_offset..block_offset + HEAD_BLOCK_SIZE],
                        &mut self.tail_output0[block_offset..block_offset + HEAD_BLOCK_SIZE],
                    );
                    if self.tail_input_fill == TAIL_BLOCK_SIZE {
                        std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                    }
                }

                // Convolution: 2nd-Nth tail block (might be done in some background thread)
                if self.tail_precalculated.len() > 0
                    && self.tail_input_fill == TAIL_BLOCK_SIZE
                    && self.tail_output.len() == TAIL_BLOCK_SIZE
                {
                    std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                    self.tail_convolver
                        .process(&self.tail_input, &mut self.tail_output);
                }

                if self.tail_input_fill == TAIL_BLOCK_SIZE {
                    self.tail_input_fill = 0;
                    self.precalculated_pos = 0;
                }

                processed += processing;
            }
        }
    }
}

#[cfg(feature = "ipp")]
mod ipp_fft {
    use crate::{Convolution, Sample};
    use ipp_sys::*;
    use num_complex::Complex32;
    use std::mem;
    use std::ptr;
    use std::slice;

    pub struct Fft {
        size: usize,
        spec: Vec<u8>,
        scratch_buffer: Vec<u8>,
    }

    impl Default for Fft {
        fn default() -> Self {
            Self {
                size: 0,
                spec: Vec::new(),
                scratch_buffer: Vec::new(),
            }
        }
    }

    impl std::fmt::Debug for Fft {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "IppFft(size: {})", self.size)
        }
    }

    impl Clone for Fft {
        fn clone(&self) -> Self {
            let mut new_fft = Fft::default();
            if self.size > 0 {
                new_fft.init(self.size);
            }
            new_fft
        }
    }

    impl Fft {
        pub fn new(size: usize) -> Self {
            let mut fft = Self::default();
            fft.init(size);
            fft
        }

        pub fn init(&mut self, size: usize) {
            // Clean up old resources if any
            self.clean_up();

            if size == 0 {
                return;
            }

            assert!(size.is_power_of_two(), "FFT size must be a power of 2");

            let mut spec_size = 0;
            let mut init_size = 0;
            let mut work_buf_size = 0;

            unsafe {
                ippsDFTGetSize_R_32f(
                    size as i32,
                    8, // IPP_FFT_NODIV_BY_ANY
                    0, // No special hint
                    &mut spec_size,
                    &mut init_size,
                    &mut work_buf_size,
                );
            }

            // Allocate aligned memory for buffers to get best SIMD performance
            let mut init_buffer = allocate_aligned_buffer(init_size as usize);
            let scratch_buffer = allocate_aligned_buffer(work_buf_size as usize);
            let mut spec = allocate_aligned_buffer(spec_size as usize);

            unsafe {
                ippsDFTInit_R_32f(
                    size as i32,
                    8, // IPP_FFT_NODIV_BY_ANY
                    0, // No special hint
                    spec.as_mut_ptr() as *mut ipp_sys::IppsDFTSpec_R_32f,
                    init_buffer.as_mut_ptr(),
                );
            }

            // Free the init buffer as it's no longer needed
            unsafe {
                if !init_buffer.is_empty() {
                    let ptr = init_buffer.as_mut_ptr();
                    mem::forget(mem::replace(&mut init_buffer, Vec::new()));
                    ippsFree(ptr as *mut _);
                }
            }

            self.size = size;
            self.spec = spec;
            self.scratch_buffer = scratch_buffer;
        }

        pub fn forward(
            &mut self,
            input: &mut [f32],
            output: &mut [Complex32],
        ) -> Result<(), &'static str> {
            if self.size == 0 {
                return Err("FFT not initialized");
            }

            assert_eq!(input.len(), self.size, "Input length must match FFT size");
            assert_eq!(
                output.len(),
                self.size / 2 + 1,
                "Output length must be size/2 + 1"
            );

            unsafe {
                ippsDFTFwd_RToCCS_32f(
                    input.as_ptr(),
                    output.as_mut_ptr() as *mut _,
                    self.spec.as_ptr() as *const DFTSpec_R_32f,
                    self.scratch_buffer.as_mut_ptr(),
                );
            }

            Ok(())
        }

        pub fn inverse(
            &mut self,
            input: &mut [Complex32],
            output: &mut [f32],
        ) -> Result<(), &'static str> {
            if self.size == 0 {
                return Err("FFT not initialized");
            }

            assert_eq!(
                input.len(),
                self.size / 2 + 1,
                "Input length must be size/2 + 1"
            );
            assert_eq!(output.len(), self.size, "Output length must match FFT size");

            unsafe {
                ippsDFTInv_CCSToR_32f(
                    input.as_ptr() as *const _,
                    output.as_mut_ptr(),
                    self.spec.as_ptr() as *const DFTSpec_R_32f,
                    self.scratch_buffer.as_mut_ptr(),
                );
            }

            // Normalize the result
            unsafe {
                ippsDivC_32f_I(self.size as f32, output.as_mut_ptr(), self.size as i32);
            }

            Ok(())
        }

        fn clean_up(&mut self) {
            unsafe {
                if !self.spec.is_empty() {
                    let ptr = self.spec.as_mut_ptr();
                    mem::forget(mem::replace(&mut self.spec, Vec::new()));
                    ippsFree(ptr as *mut _);
                }

                if !self.scratch_buffer.is_empty() {
                    let ptr = self.scratch_buffer.as_mut_ptr();
                    mem::forget(mem::replace(&mut self.scratch_buffer, Vec::new()));
                    ippsFree(ptr as *mut _);
                }
            }
            self.size = 0;
        }
    }

    // Drop trait implementation to free aligned memory
    impl Drop for Fft {
        fn drop(&mut self) {
            self.clean_up();
        }
    }

    // Helper function to allocate aligned memory using IPP functions
    fn allocate_aligned_buffer(size: usize) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }

        unsafe {
            let ptr = ippsMalloc_8u(size as i32);
            if ptr.is_null() {
                panic!("Failed to allocate aligned memory");
            }

            // Construct a vector from the IPP-aligned pointer
            Vec::from_raw_parts(ptr, size, size)
        }
    }

    // Helper functions for FFT convolution
    pub fn complex_size(size: usize) -> usize {
        (size / 2) + 1
    }

    pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
        assert!(dst.len() >= src_size, "Destination buffer too small");

        // Copy the source data
        unsafe {
            ippsCopy_32f(src.as_ptr(), dst.as_mut_ptr(), src_size as i32);

            // Zero-pad the rest
            if dst.len() > src_size {
                ippsZero_32f(
                    dst.as_mut_ptr().add(src_size),
                    (dst.len() - src_size) as i32,
                );
            }
        }
    }

    pub fn complex_multiply_accumulate(
        result: &mut [Complex32],
        a: &[Complex32],
        b: &[Complex32],
        temp_buffer: &mut [Complex32],
    ) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());
        assert_eq!(temp_buffer.len(), a.len());

        unsafe {
            // Use pre-allocated temp buffer instead of allocating
            let len = result.len();

            // Use ippsMul_32fc instead of ippsMulC_32fc to correctly multiply arrays
            ippsMul_32fc(
                a.as_ptr() as *const ipp_sys::Ipp32fc,
                b.as_ptr() as *const ipp_sys::Ipp32fc,
                temp_buffer.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                len as i32,
            );

            ippsAdd_32fc(
                temp_buffer.as_ptr() as *const ipp_sys::Ipp32fc,
                result.as_ptr() as *const ipp_sys::Ipp32fc,
                result.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                len as i32,
            );
        }
    }

    pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());

        unsafe {
            ippsAdd_32f(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                result.len() as i32,
            );
        }
    }
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
    #[derive(Default, Clone)]
    pub struct FFTConvolver {
        ir_len: usize,
        block_size: usize,
        seg_size: usize,
        seg_count: usize,
        active_seg_count: usize,
        fft_complex_size: usize,
        segments: Vec<Vec<Complex32>>,
        segments_ir: Vec<Vec<Complex32>>,
        fft_buffer: Vec<f32>,
        fft: Fft,
        pre_multiplied: Vec<Complex32>,
        conv: Vec<Complex32>,
        overlap: Vec<f32>,
        current: usize,
        input_buffer: Vec<f32>,
        input_buffer_fill: usize,
        temp_buffer: Vec<Complex32>,
    }

    impl Convolution for FFTConvolver {
        fn init(
            impulse_response: &[Sample],
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
            let fft_complex_size = complex_size(seg_size);

            // FFT
            let mut fft = Fft::default();
            fft.init(seg_size);
            let mut fft_buffer = vec![0.; seg_size];

            // prepare segments
            let mut segments = Vec::new();
            for _ in 0..seg_count {
                segments.push(vec![Complex32::new(0., 0.); fft_complex_size]);
            }

            let mut segments_ir = Vec::new();

            // prepare ir
            for i in 0..seg_count {
                let mut segment = vec![Complex32::new(0., 0.); fft_complex_size];
                let remaining = ir_len - (i * block_size);
                let size_copy = if remaining >= block_size {
                    block_size
                } else {
                    remaining
                };
                copy_and_pad(&mut fft_buffer, &padded_ir[i * block_size..], size_copy);
                fft.forward(&mut fft_buffer, &mut segment).unwrap();
                segments_ir.push(segment);
            }

            // prepare convolution buffers
            let pre_multiplied = vec![Complex32::new(0., 0.); fft_complex_size];
            let conv = vec![Complex32::new(0., 0.); fft_complex_size];
            let overlap = vec![0.; block_size];

            // prepare input buffer
            let input_buffer = vec![0.; block_size];
            let input_buffer_fill = 0;

            // reset current position
            let current = 0;
            let temp_buffer = vec![Complex32::new(0.0, 0.0); fft_complex_size];

            Self {
                ir_len,
                block_size,
                seg_size,
                seg_count,
                active_seg_count,
                fft_complex_size,
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
                temp_buffer,
            }
        }

        fn update(&mut self, response: &[Sample]) {
            let new_ir_len = response.len();

            if new_ir_len > self.ir_len {
                panic!("New impulse response is longer than initialized length");
            }

            if self.ir_len == 0 {
                return;
            }

            unsafe {
                // Zero out buffers
                ippsZero_32f(self.fft_buffer.as_mut_ptr(), self.fft_buffer.len() as i32);
                ippsZero_32fc(
                    self.conv.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                    self.conv.len() as i32,
                );
                ippsZero_32fc(
                    self.pre_multiplied.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                    self.pre_multiplied.len() as i32,
                );
                ippsZero_32f(self.overlap.as_mut_ptr(), self.overlap.len() as i32);
            }

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
                copy_and_pad(
                    &mut self.fft_buffer,
                    &response[i * self.block_size..],
                    size_copy,
                );
                self.fft.forward(&mut self.fft_buffer, segment).unwrap();
            }

            // Clear remaining segments
            for i in self.active_seg_count..self.seg_count {
                unsafe {
                    ippsZero_32fc(
                        self.segments_ir[i].as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                        self.segments_ir[i].len() as i32,
                    );
                }
            }
        }

        fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
            if self.active_seg_count == 0 {
                unsafe {
                    ippsZero_32f(output.as_mut_ptr(), output.len() as i32);
                }
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
                unsafe {
                    ippsCopy_32f(
                        &input[processed],
                        &mut self.input_buffer[input_buffer_pos],
                        processing as i32,
                    );
                }

                // Forward FFT
                copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
                if let Err(_) = self
                    .fft
                    .forward(&mut self.fft_buffer, &mut self.segments[self.current])
                {
                    unsafe {
                        ippsZero_32f(output.as_mut_ptr(), output.len() as i32);
                    }
                    return;
                }

                // Complex multiplication
                if input_buffer_was_empty {
                    unsafe {
                        ippsZero_32fc(
                            self.pre_multiplied.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                            self.pre_multiplied.len() as i32,
                        );
                    }

                    for i in 1..self.active_seg_count {
                        let index_ir = i;
                        let index_audio = (self.current + i) % self.active_seg_count;
                        complex_multiply_accumulate(
                            &mut self.pre_multiplied,
                            &self.segments_ir[index_ir],
                            &self.segments[index_audio],
                            &mut self.temp_buffer,
                        );
                    }
                }

                // Copy pre-multiplied to conv
                unsafe {
                    ippsCopy_32fc(
                        self.pre_multiplied.as_ptr() as *const ipp_sys::Ipp32fc,
                        self.conv.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                        self.conv.len() as i32,
                    );
                }

                complex_multiply_accumulate(
                    &mut self.conv,
                    &self.segments[self.current],
                    &self.segments_ir[0],
                    &mut self.temp_buffer,
                );

                // Backward FFT
                if let Err(_) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                    unsafe {
                        ippsZero_32f(output.as_mut_ptr(), output.len() as i32);
                    }
                    return;
                }

                // Add overlap
                sum(
                    &mut output[processed..processed + processing],
                    &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                    &self.overlap[input_buffer_pos..input_buffer_pos + processing],
                );

                // Input buffer full => Next block
                self.input_buffer_fill += processing;
                if self.input_buffer_fill == self.block_size {
                    // Input buffer is empty again now
                    unsafe {
                        ippsZero_32f(
                            self.input_buffer.as_mut_ptr(),
                            self.input_buffer.len() as i32,
                        );
                    }
                    self.input_buffer_fill = 0;

                    // Save the overlap
                    unsafe {
                        ippsCopy_32f(
                            &self.fft_buffer[self.block_size],
                            self.overlap.as_mut_ptr(),
                            self.block_size as i32,
                        );
                    }

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

    #[derive(Clone)]
    pub struct TwoStageFFTConvolver {
        head_convolver: FFTConvolver,
        tail_convolver0: FFTConvolver,
        tail_output0: Vec<Sample>,
        tail_precalculated0: Vec<Sample>,
        tail_convolver: FFTConvolver,
        tail_output: Vec<Sample>,
        tail_precalculated: Vec<Sample>,
        tail_input: Vec<Sample>,
        tail_input_fill: usize,
        precalculated_pos: usize,
    }

    const HEAD_BLOCK_SIZE: usize = 128;
    const TAIL_BLOCK_SIZE: usize = 1024;

    impl Convolution for TwoStageFFTConvolver {
        fn init(
            impulse_response: &[Sample],
            _block_size: usize,
            max_response_length: usize,
        ) -> Self {
            let head_block_size = HEAD_BLOCK_SIZE;
            let tail_block_size = TAIL_BLOCK_SIZE;

            if max_response_length < impulse_response.len() {
                panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
            }
            let mut padded_ir = impulse_response.to_vec();
            padded_ir.resize(max_response_length, 0.);

            let head_ir_len = std::cmp::min(max_response_length, tail_block_size);
            let head_convolver = FFTConvolver::init(
                &padded_ir[0..head_ir_len],
                head_block_size,
                max_response_length,
            );

            let tail_convolver0 = (max_response_length > tail_block_size)
                .then(|| {
                    let tail_ir_len =
                        std::cmp::min(max_response_length - tail_block_size, tail_block_size);
                    FFTConvolver::init(
                        &padded_ir[tail_block_size..tail_block_size + tail_ir_len],
                        head_block_size,
                        max_response_length,
                    )
                })
                .unwrap_or_default();

            let tail_output0 = vec![0.0; tail_block_size];
            let tail_precalculated0 = vec![0.0; tail_block_size];

            let tail_convolver = (max_response_length > 2 * tail_block_size)
                .then(|| {
                    let tail_ir_len = max_response_length - 2 * tail_block_size;
                    FFTConvolver::init(
                        &padded_ir[2 * tail_block_size..2 * tail_block_size + tail_ir_len],
                        tail_block_size,
                        max_response_length,
                    )
                })
                .unwrap_or_default();

            let tail_output = vec![0.0; tail_block_size];
            let tail_precalculated = vec![0.0; tail_block_size];
            let tail_input = vec![0.0; tail_block_size];
            let tail_input_fill = 0;
            let precalculated_pos = 0;

            TwoStageFFTConvolver {
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
            }
        }

        fn update(&mut self, _response: &[Sample]) {
            todo!()
        }

        fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
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
                    HEAD_BLOCK_SIZE - (self.tail_input_fill % HEAD_BLOCK_SIZE),
                );

                // Sum head and tail
                let sum_begin = processed;
                let sum_end = processed + processing;

                // Sum: 1st tail block
                if !self.tail_precalculated0.is_empty() {
                    let mut precalculated_pos = self.precalculated_pos;

                    // Use IPP to add the tail block to output
                    unsafe {
                        ippsAdd_32f(
                            &self.tail_precalculated0[precalculated_pos],
                            &output[sum_begin],
                            &mut output[sum_begin],
                            processing as i32,
                        );
                        // ippsAddC_32f_I(
                        //     std::slice::from_raw_parts(
                        //         self.tail_precalculated0.as_ptr().add(precalculated_pos),
                        //         processing,
                        //     )
                        //     .as_ptr(),
                        //     output[sum_begin..sum_end].as_mut_ptr(),
                        //     processing as i32,
                        // );
                    }
                }

                // Sum: 2nd-Nth tail block
                if !self.tail_precalculated.is_empty() {
                    let precalculated_pos = self.precalculated_pos;

                    // Use IPP to add the tail block to output
                    unsafe {
                        // ippsAddC_32f_I(
                        //     std::slice::from_raw_parts(
                        //         self.tail_precalculated.as_ptr().add(precalculated_pos),
                        //         processing,
                        //     )
                        //     .as_ptr(),
                        //     output[sum_begin..sum_end].as_mut_ptr(),
                        //     processing as i32,
                        // );
                        ippsAdd_32f(
                            &self.tail_precalculated[precalculated_pos],
                            &output[sum_begin],
                            &mut output[sum_begin],
                            processing as i32,
                        );
                    }
                }

                self.precalculated_pos += processing;

                // Fill input buffer for tail convolution
                unsafe {
                    ippsCopy_32f(
                        &input[processed],
                        &mut self.tail_input[self.tail_input_fill],
                        processing as i32,
                    );
                }
                self.tail_input_fill += processing;

                // Convolution: 1st tail block
                if !self.tail_precalculated0.is_empty()
                    && self.tail_input_fill % HEAD_BLOCK_SIZE == 0
                {
                    assert!(self.tail_input_fill >= HEAD_BLOCK_SIZE);
                    let block_offset = self.tail_input_fill - HEAD_BLOCK_SIZE;
                    self.tail_convolver0.process(
                        &self.tail_input[block_offset..block_offset + HEAD_BLOCK_SIZE],
                        &mut self.tail_output0[block_offset..block_offset + HEAD_BLOCK_SIZE],
                    );
                    if self.tail_input_fill == TAIL_BLOCK_SIZE {
                        std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                    }
                }

                // Convolution: 2nd-Nth tail block (might be done in some background thread)
                if !self.tail_precalculated.is_empty()
                    && self.tail_input_fill == TAIL_BLOCK_SIZE
                    && self.tail_output.len() == TAIL_BLOCK_SIZE
                {
                    std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                    self.tail_convolver
                        .process(&self.tail_input, &mut self.tail_output);
                }

                if self.tail_input_fill == TAIL_BLOCK_SIZE {
                    self.tail_input_fill = 0;
                    self.precalculated_pos = 0;
                }

                processed += processing;
            }
        }
    }
}

#[cfg(all(test, feature = "ipp"))]
mod tests {
    use super::*;
    use crate::{fft_convolver::rust_fft, Convolution};

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

#[cfg(not(feature = "ipp"))]
pub use rust_fft::{
    complex_multiply_accumulate, complex_size, copy_and_pad, sum, FFTConvolver, Fft,
    TwoStageFFTConvolver,
};

#[cfg(feature = "ipp")]
pub use self::ipp_fft::{
    complex_multiply_accumulate, complex_size, copy_and_pad, sum, FFTConvolver, Fft,
    TwoStageFFTConvolver,
};
