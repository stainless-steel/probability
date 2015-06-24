use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn cdf(bencher: &mut Bencher) {
    let mut generator = generator();
    let gaussian = Gaussian::new(0.0, 1.0);
    let x = Sampler(&gaussian, &mut generator).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| gaussian.cdf(x)).collect::<Vec<_>>()));
}

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let mut generator = generator();
    let gaussian = Gaussian::new(0.0, 1.0);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Sampler(&uniform, &mut generator).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| gaussian.inv_cdf(p)).collect::<Vec<_>>()));
}

#[bench]
fn sample(bencher: &mut Bencher) {
    use probability::generator::XorshiftPlus;

    let mut generator = XorshiftPlus::new([42, 42]);
    let gaussian = Gaussian::new(0.0, 1.0);

    bencher.iter(|| black_box(gaussian.sample(&mut generator)));
}
