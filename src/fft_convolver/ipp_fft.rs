use crate::{Convolution, Sample};
use ipp_sys::*;
use num_complex::Complex32;
use std::mem;

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
        self.clean_up();

        assert!(size > 0, "FFT size must be greater than 0");
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

        let mut init_buffer = allocate_aligned_byte_buffer(init_size as usize);
        let scratch_buffer = allocate_aligned_byte_buffer(work_buf_size as usize);
        let mut spec = allocate_aligned_byte_buffer(spec_size as usize);

        unsafe {
            ippsDFTInit_R_32f(
                size as i32,
                8, // IPP_FFT_NODIV_BY_ANY
                0, // No special hint
                spec.as_mut_ptr() as *mut ipp_sys::IppsDFTSpec_R_32f,
                init_buffer.as_mut_ptr(),
            );
        }

        drop_aligned_buffer(&mut init_buffer);

        self.size = size;
        self.spec = spec;
        self.scratch_buffer = scratch_buffer;
    }

    pub fn forward(&mut self, input: &[f32], output: &mut [Complex32]) -> Result<(), &'static str> {
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
                output.as_mut_ptr() as *mut Ipp32f, // the API takes the pointer to the first float member of the complex array
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
                input.as_ptr() as *const Ipp32f, // the API takes the pointer to the first float member of the complex array
                output.as_mut_ptr(),
                self.spec.as_ptr() as *const DFTSpec_R_32f,
                self.scratch_buffer.as_mut_ptr(),
            );
        }

        // Normalization
        unsafe {
            ippsDivC_32f_I(self.size as f32, output.as_mut_ptr(), self.size as i32);
        }

        Ok(())
    }

    fn clean_up(&mut self) {
        drop_aligned_buffer(&mut self.spec);
        drop_aligned_buffer(&mut self.scratch_buffer);
        self.size = 0;
    }
}

impl Drop for Fft {
    fn drop(&mut self) {
        self.clean_up();
    }
}

// Helper functions for allocating/dropping/cloning aligned memory using IPP functions
fn allocate_aligned_byte_buffer(size: usize) -> Vec<u8> {
    if size == 0 {
        return Vec::new();
    }

    unsafe {
        let ptr = ippsMalloc_8u(size as i32);
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory");
        }
        ippsZero_8u(ptr, size as i32);
        Vec::from_raw_parts(ptr, size, size)
    }
}

fn allocate_aligned_complex_f32_buffer(size: usize) -> Vec<Complex32> {
    if size == 0 {
        return Vec::new();
    }

    unsafe {
        let ptr = ippsMalloc_32fc(size as i32);
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory for complex buffer");
        }
        ippsZero_32fc(ptr as *mut ipp_sys::Ipp32fc, size as i32);
        Vec::from_raw_parts(ptr as *mut Complex32, size, size)
    }
}

fn allocate_aligned_f32_buffer(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }

    unsafe {
        let ptr = ippsMalloc_32f(size as i32);
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory for float buffer");
        }
        ippsZero_32f(ptr, size as i32);
        Vec::from_raw_parts(ptr as *mut f32, size, size)
    }
}

fn drop_aligned_buffer<T>(buffer: &mut Vec<T>) {
    unsafe {
        if !buffer.is_empty() {
            let ptr = buffer.as_mut_ptr();
            mem::forget(mem::replace(buffer, Vec::new()));
            ippsFree(ptr as *mut _);
        }
    }
}

fn clone_aligned_f32_buffer(src: &[f32]) -> Vec<f32> {
    if src.is_empty() {
        return Vec::new();
    }

    let mut new_buffer = allocate_aligned_f32_buffer(src.len());
    unsafe {
        ippsCopy_32f(src.as_ptr(), new_buffer.as_mut_ptr(), src.len() as i32);
    }
    new_buffer
}

fn clone_aligned_complex_f32_buffer(src: &[Complex32]) -> Vec<Complex32> {
    if src.is_empty() {
        return Vec::new();
    }

    let mut new_buffer = allocate_aligned_complex_f32_buffer(src.len());
    unsafe {
        ippsCopy_32fc(
            src.as_ptr() as *const ipp_sys::Ipp32fc,
            new_buffer.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
            src.len() as i32,
        );
    }
    new_buffer
}

pub fn aligned_f32_swap(src: &mut [f32], dst: &mut [f32], temp: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "Buffers must be the same length for swapping"
    );

    unsafe {
        ippsCopy_32f(src.as_ptr(), temp.as_mut_ptr(), src.len() as i32);
        ippsCopy_32f(dst.as_ptr(), src.as_mut_ptr(), dst.len() as i32);
        ippsCopy_32f(temp.as_ptr(), dst.as_mut_ptr(), temp.len() as i32);
    }
}

pub fn complex_size(size: usize) -> usize {
    (size / 2) + 1
}

pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
    assert!(dst.len() >= src_size, "Destination buffer too small");

    unsafe {
        ippsCopy_32f(src.as_ptr(), dst.as_mut_ptr(), src_size as i32);

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
        let len = result.len() as i32;

        ippsMul_32fc(
            a.as_ptr() as *const ipp_sys::Ipp32fc,
            b.as_ptr() as *const ipp_sys::Ipp32fc,
            temp_buffer.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
            len,
        );

        ippsAdd_32fc(
            temp_buffer.as_ptr() as *const ipp_sys::Ipp32fc,
            result.as_ptr() as *const ipp_sys::Ipp32fc,
            result.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
            len,
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

#[derive(Default)]
pub struct FFTConvolver {
    ir_len: usize,
    block_size: usize,
    _seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    _fft_complex_size: usize,
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
    fn init(impulse_response: &[Sample], block_size: usize, max_response_length: usize) -> Self {
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
        let mut fft_buffer = allocate_aligned_f32_buffer(seg_size);

        // prepare segments with aligned vectors
        let mut segments = Vec::new();
        for _ in 0..seg_count {
            segments.push(allocate_aligned_complex_f32_buffer(fft_complex_size));
        }

        let mut segments_ir = Vec::new();

        // prepare ir
        for i in 0..seg_count {
            let mut segment = allocate_aligned_complex_f32_buffer(fft_complex_size);
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

        // prepare convolution buffers with aligned memory
        let pre_multiplied = allocate_aligned_complex_f32_buffer(fft_complex_size);
        let conv = allocate_aligned_complex_f32_buffer(fft_complex_size);
        let overlap = allocate_aligned_f32_buffer(block_size);

        // prepare input buffer
        let input_buffer = allocate_aligned_f32_buffer(block_size);
        let input_buffer_fill = 0;

        // reset current position
        let current = 0;
        let temp_buffer = allocate_aligned_complex_f32_buffer(fft_complex_size);

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

            ippsZero_32f(
                self.input_buffer.as_mut_ptr(),
                self.input_buffer.len() as i32,
            );
        }

        self.input_buffer_fill = 0;
        self.current = 0;

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

        for segment in &mut self.segments {
            unsafe {
                ippsZero_32fc(
                    segment.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                    segment.len() as i32,
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

impl Clone for FFTConvolver {
    fn clone(&self) -> Self {
        let mut segments = Vec::new();
        for segment in &self.segments {
            segments.push(clone_aligned_complex_f32_buffer(segment));
        }

        let mut segments_ir = Vec::new();
        for segment_ir in &self.segments_ir {
            segments_ir.push(clone_aligned_complex_f32_buffer(segment_ir));
        }

        let fft_buffer = clone_aligned_f32_buffer(&self.fft_buffer);
        let pre_multiplied = clone_aligned_complex_f32_buffer(&self.pre_multiplied);
        let conv = clone_aligned_complex_f32_buffer(&self.conv);
        let overlap = clone_aligned_f32_buffer(&self.overlap);
        let input_buffer = clone_aligned_f32_buffer(&self.input_buffer);
        let temp_buffer = clone_aligned_complex_f32_buffer(&self.temp_buffer);

        Self {
            ir_len: self.ir_len,
            block_size: self.block_size,
            _seg_size: self._seg_size,
            seg_count: self.seg_count,
            active_seg_count: self.active_seg_count,
            _fft_complex_size: self._fft_complex_size,
            segments,
            segments_ir,
            fft_buffer,
            fft: self.fft.clone(),
            pre_multiplied,
            conv,
            overlap,
            current: self.current,
            input_buffer,
            input_buffer_fill: self.input_buffer_fill,
            temp_buffer,
        }
    }
}

impl Drop for FFTConvolver {
    fn drop(&mut self) {
        for segment in &mut self.segments {
            drop_aligned_buffer(segment);
        }

        for segment_ir in &mut self.segments_ir {
            drop_aligned_buffer(segment_ir);
        }

        drop_aligned_buffer(&mut self.fft_buffer);
        drop_aligned_buffer(&mut self.pre_multiplied);
        drop_aligned_buffer(&mut self.conv);
        drop_aligned_buffer(&mut self.overlap);
        drop_aligned_buffer(&mut self.input_buffer);
        drop_aligned_buffer(&mut self.temp_buffer);
    }
}

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
    swap_buffer: Vec<Sample>,
}

const HEAD_BLOCK_SIZE: usize = 128;
const TAIL_BLOCK_SIZE: usize = 1024;

impl Convolution for TwoStageFFTConvolver {
    fn init(impulse_response: &[Sample], _block_size: usize, max_response_length: usize) -> Self {
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

        let tail_output0 = allocate_aligned_f32_buffer(tail_block_size);
        let tail_precalculated0 = allocate_aligned_f32_buffer(tail_block_size);

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

        let tail_output = allocate_aligned_f32_buffer(tail_block_size);
        let tail_precalculated = allocate_aligned_f32_buffer(tail_block_size);
        let tail_input = allocate_aligned_f32_buffer(tail_block_size);
        let swap_buffer = allocate_aligned_f32_buffer(tail_block_size);

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
            swap_buffer,
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

            // Sum: 1st tail block
            if !self.tail_precalculated0.is_empty() {
                let precalculated_pos = self.precalculated_pos;

                // Use IPP to add the tail block to output
                unsafe {
                    ippsAdd_32f(
                        &self.tail_precalculated0[precalculated_pos],
                        &output[sum_begin],
                        &mut output[sum_begin],
                        processing as i32,
                    );
                }
            }

            // Sum: 2nd-Nth tail block
            if !self.tail_precalculated.is_empty() {
                let precalculated_pos = self.precalculated_pos;

                // Use IPP to add the tail block to output
                unsafe {
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
            if !self.tail_precalculated0.is_empty() && self.tail_input_fill % HEAD_BLOCK_SIZE == 0 {
                assert!(self.tail_input_fill >= HEAD_BLOCK_SIZE);
                let block_offset = self.tail_input_fill - HEAD_BLOCK_SIZE;
                self.tail_convolver0.process(
                    &self.tail_input[block_offset..block_offset + HEAD_BLOCK_SIZE],
                    &mut self.tail_output0[block_offset..block_offset + HEAD_BLOCK_SIZE],
                );
                if self.tail_input_fill == TAIL_BLOCK_SIZE {
                    aligned_f32_swap(
                        &mut self.tail_precalculated0,
                        &mut self.tail_output0,
                        &mut self.swap_buffer,
                    );
                }
            }

            // Convolution: 2nd-Nth tail block (might be done in some background thread)
            if !self.tail_precalculated.is_empty()
                && self.tail_input_fill == TAIL_BLOCK_SIZE
                && self.tail_output.len() == TAIL_BLOCK_SIZE
            {
                aligned_f32_swap(
                    &mut self.tail_precalculated,
                    &mut self.tail_output,
                    &mut self.swap_buffer,
                );
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

impl Clone for TwoStageFFTConvolver {
    fn clone(&self) -> Self {
        let head_convolver = self.head_convolver.clone();
        let tail_convolver0 = self.tail_convolver0.clone();
        let tail_convolver = self.tail_convolver.clone();

        let tail_output0 = clone_aligned_f32_buffer(&self.tail_output0);
        let tail_precalculated0 = clone_aligned_f32_buffer(&self.tail_precalculated0);
        let tail_output = clone_aligned_f32_buffer(&self.tail_output);
        let tail_precalculated = clone_aligned_f32_buffer(&self.tail_precalculated);
        let tail_input = clone_aligned_f32_buffer(&self.tail_input);
        let swap_buffer = clone_aligned_f32_buffer(&self.swap_buffer);

        Self {
            head_convolver,
            tail_convolver0,
            tail_output0,
            tail_precalculated0,
            tail_convolver,
            tail_output,
            tail_precalculated,
            tail_input,
            tail_input_fill: self.tail_input_fill,
            precalculated_pos: self.precalculated_pos,
            swap_buffer,
        }
    }
}

impl Drop for TwoStageFFTConvolver {
    fn drop(&mut self) {
        drop_aligned_buffer(&mut self.tail_output0);
        drop_aligned_buffer(&mut self.tail_precalculated0);
        drop_aligned_buffer(&mut self.tail_output);
        drop_aligned_buffer(&mut self.tail_precalculated);
        drop_aligned_buffer(&mut self.tail_input);
        drop_aligned_buffer(&mut self.swap_buffer);
    }
}
