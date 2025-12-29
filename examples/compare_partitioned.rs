mod util;

use convolution::Convolution;
use convolution::fft_convolver::*;
use util::*;

const SAMPLE_RATE: u32 = 44100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let block_size = 64;
    let n_blocks = 1000;

    let response = generate_sinusoid(65000, 1000.0, SAMPLE_RATE, 0.1);

    let mut convolver_a = FFTConvolver::init(&response, block_size, response.len());
    let mut convolver_b = TwoStageFFTConvolver::init(&response, block_size, response.len());

    let mut output_a = vec![0.0; block_size * n_blocks];
    let mut output_b = vec![0.0; block_size * n_blocks];

    let mut block_a = vec![0.0; block_size];
    let mut block_b = vec![0.0; block_size];

    let input = generate_sinusoid(n_blocks * block_size, 1300.0, SAMPLE_RATE, 0.1);

    // for i in 0..n_blocks {
    //     let start = i * block_size;
    //     let end = (i + 1) * block_size;
    //     convolver_a.process(&input[start..end], &mut block_a);
    //     convolver_b.process(&input[start..end], &mut block_b);

    //     output_a[start..end].copy_from_slice(&block_a);
    //     output_b[start..end].copy_from_slice(&block_b);
    // }

    let time = std::time::Instant::now();

    for i in 0..n_blocks {
        let start = i * block_size;
        let end = (i + 1) * block_size;
        convolver_a.process(&input[start..end], &mut block_a);
        output_a[start..end].copy_from_slice(&block_a);
    }
    println!(
        "Uniform took = {:.2} ms",
        time.elapsed().as_secs_f64() * 1000.0
    );

    let time = std::time::Instant::now();

    for i in 0..n_blocks {
        let start = i * block_size;
        let end = (i + 1) * block_size;
        convolver_b.process(&input[start..end], &mut block_b);
        output_b[start..end].copy_from_slice(&block_b);
    }

    println!(
        "Partitioned took = {:.2} ms",
        time.elapsed().as_secs_f64() * 1000.0
    );

    let mut max_abs_diff = 0.;
    for (a, b) in std::iter::zip(&output_a, &output_b) {
        let abs_diff = (a - b).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
    }
    println!("max_abs_diff = {:?}", max_abs_diff);

    save_wav("output_a.wav", &output_a, SAMPLE_RATE as u32)?;
    save_wav("output_b.wav", &output_b, SAMPLE_RATE as u32)?;

    Ok(())
}
