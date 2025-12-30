use realfft::{ComplexToReal, FftError, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

use crate::Convolution;

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

    pub fn forward(&self, input: &mut [f32], output: &mut [Complex<f32>]) -> Result<(), FftError> {
        self.fft_forward.process(input, output)?;
        Ok(())
    }

    pub fn inverse(&self, input: &mut [Complex<f32>], output: &mut [f32]) -> Result<(), FftError> {
        self.fft_inverse.process(input, output)?;

        // FFT Normalization
        let len = output.len();
        output.iter_mut().for_each(|bin| *bin /= len as f32);

        Ok(())
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

#[inline]
pub fn complex_multiply_accumulate(
    result: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    for i in 0..len {
        result[i] += a[i] * b[i];
    }
}

#[inline]
pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    for i in 0..len {
        result[i] = a[i] + b[i];
    }
}

#[derive(Default, Clone)]
pub struct FFTConvolver {
    ir_len: usize,
    block_size: usize,
    seg_count: usize,
    active_seg_count: usize,
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
    fn init(impulse_response: &[f32], block_size: usize, max_response_length: usize) -> Self {
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
            seg_count,
            active_seg_count,
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

    fn update(&mut self, response: &[f32]) {
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

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
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
    fn reset(&mut self) {
        self.overlap.fill(0.);
        for s in &mut self.segments {
            s.fill(Complex::new(0., 0.));
        }
        self.current = 0;
        self.input_buffer.fill(0.);
        self.pre_multiplied.fill(Complex::new(0., 0.));
        self.conv.fill(Complex::new(0., 0.));
        self.input_buffer_fill = 0;
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
    head_block_size: usize,
    tail_block_size: usize,
    head_convolver: FFTConvolver,
    tail_convolver0: FFTConvolver,
    tail_output0: Vec<f32>,
    tail_precalculated0: Vec<f32>,
    tail_convolver: FFTConvolver,
    tail_output: Vec<f32>,
    tail_precalculated: Vec<f32>,
    tail_input: Vec<f32>,
    tail_input_fill: usize,
    precalculated_pos: usize,
}

impl Convolution for TwoStageFFTConvolver {
    fn init(impulse_response: &[f32], block_size: usize, max_response_length: usize) -> Self {
        let head_block_size = block_size;
        let tail_block_size = compute_tail_block_size(block_size, max_response_length);

        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);

        let head_ir_len = std::cmp::min(max_response_length, tail_block_size);
        let head_convolver =
            FFTConvolver::init(&padded_ir[0..head_ir_len], head_block_size, head_ir_len);

        let tail_convolver0 = if max_response_length > tail_block_size {
            {
                let tail_ir_len =
                    std::cmp::min(max_response_length - tail_block_size, tail_block_size);
                FFTConvolver::init(
                    &padded_ir[tail_block_size..tail_block_size + tail_ir_len],
                    head_block_size,
                    tail_ir_len,
                )
            }
        } else {
            Default::default()
        };

        let tail_output0 = vec![0.0; tail_block_size];
        let tail_precalculated0 = vec![0.0; tail_block_size];

        let tail_convolver = if max_response_length > 2 * tail_block_size {
            {
                let tail_ir_len = max_response_length - 2 * tail_block_size;
                FFTConvolver::init(
                    &padded_ir[2 * tail_block_size..2 * tail_block_size + tail_ir_len],
                    tail_block_size,
                    tail_ir_len,
                )
            }
        } else {
            Default::default()
        };

        let tail_output = vec![0.0; tail_block_size];
        let tail_precalculated = vec![0.0; tail_block_size];
        let tail_input = vec![0.0; tail_block_size];
        let tail_input_fill = 0;
        let precalculated_pos = 0;

        TwoStageFFTConvolver {
            head_block_size,
            tail_block_size,
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

    fn update(&mut self, _response: &[f32]) {
        todo!()
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // TODO: Not sure why this is necessary here but not for FFTConvolver
        assert!(input.len() <= self.head_block_size);

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
                self.head_block_size - (self.tail_input_fill % self.head_block_size),
            );

            // Sum head and tail
            let sum_begin = processed;
            let sum_end = processed + processing;

            // Sum: 1st tail block
            if !self.tail_precalculated0.is_empty() {
                let mut precalculated_pos = self.precalculated_pos;
                for i in sum_begin..sum_end {
                    output[i] += self.tail_precalculated0[precalculated_pos];
                    precalculated_pos += 1;
                }
            }

            // Sum: 2nd-Nth tail block
            if !self.tail_precalculated.is_empty() {
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
            if !self.tail_precalculated0.is_empty()
                && self.tail_input_fill.is_multiple_of(self.head_block_size)
            {
                assert!(self.tail_input_fill >= self.head_block_size);
                let block_offset = self.tail_input_fill - self.head_block_size;
                self.tail_convolver0.process(
                    &self.tail_input[block_offset..block_offset + self.head_block_size],
                    &mut self.tail_output0[block_offset..block_offset + self.head_block_size],
                );
                if self.tail_input_fill == self.tail_block_size {
                    std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                }
            }

            // Convolution: 2nd-Nth tail block (might be done in some background thread)
            if !self.tail_precalculated.is_empty()
                && self.tail_input_fill == self.tail_block_size
                && self.tail_output.len() == self.tail_block_size
            {
                std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                self.tail_convolver
                    .process(&self.tail_input, &mut self.tail_output);
            }

            if self.tail_input_fill == self.tail_block_size {
                self.tail_input_fill = 0;
                self.precalculated_pos = 0;
            }

            processed += processing;
        }
    }

    fn reset(&mut self) {
        self.head_convolver.reset();

        self.tail_convolver0.reset();
        self.tail_output0.fill(0.);
        self.tail_precalculated0.fill(0.);

        self.tail_convolver.reset();
        self.tail_output.fill(0.);
        self.tail_precalculated.fill(0.);

        self.tail_input.fill(0.);
        self.tail_input_fill = 0;
        self.precalculated_pos = 0;
    }
}

// FFT constant k, time relative to a multiply-add operation.
// We take 1.5 as suggested by the author, RustFFT might need a different value.
const FFT_K: f32 = 1.5;

// Compute optimal two-stage partition following:
// Guillermo García "Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay"
fn compute_tail_block_size(head_len: usize, response_len: usize) -> usize {
    let kn = (FFT_K * head_len as f32) / (2.0 * f32::ln(2.0));
    let b = -kn + f32::sqrt(kn * kn + (response_len as f32) * (head_len as f32));
    let b = b.max(head_len as f32);

    usize::next_power_of_two(b as usize)
}

#[test]
fn test_fft_twostage_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = TwoStageFFTConvolver::init(&response, 1024, response.len());
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
