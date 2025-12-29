#![allow(dead_code)]
#![allow(unused)]

use hound::{WavSpec, WavWriter};
use std::f64::consts::PI;

pub fn generate_sinusoid(num_samples: usize, freq: f64, sample_rate: u32, gain: f64) -> Vec<f32> {
    use std::f64::consts::PI;

    let mut out = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = (i as f64) / (sample_rate as f64);
        let x = gain * (2.0 * PI * freq * t).sin();
        out.push(x as f32);
    }

    out
}

pub fn save_wav(filename: &str, samples: &[f32], sample_rate: u32) -> Result<(), hound::Error> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec)?;

    for &sample in samples {
        // Convert f32 [-1.0, 1.0] to i16
        let scaled = (sample * (i16::MAX as f32)) as i16;
        writer.write_sample(scaled)?;
    }

    writer.finalize()?;
    println!("Saved: {}", filename);
    Ok(())
}
