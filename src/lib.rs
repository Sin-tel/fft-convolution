pub mod crossfade_convolver;
pub mod fft_convolver;
mod tests;

pub trait Convolution: Clone {
    fn init(response: &[f32], max_block_size: usize, max_response_length: usize) -> Self;

    // must be implemented in a real-time safe way, e.g. no heap allocations
    fn update(&mut self, response: &[f32]);

    fn reset(&mut self);

    fn process(&mut self, input: &[f32], output: &mut [f32]);
}
