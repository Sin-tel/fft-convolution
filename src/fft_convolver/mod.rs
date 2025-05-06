#[cfg(feature = "ipp")]
pub mod ipp_fft;
pub mod rust_fft;

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
