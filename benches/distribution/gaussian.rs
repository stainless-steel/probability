use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn cdf(bencher: &mut Bencher) {
    let mut source = random::default();
    let gaussian = Gaussian::new(0.0, 1.0);
    let x = Independent(&gaussian, &mut source).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| gaussian.cdf(x)).collect::<Vec<_>>()));
}

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let mut source = random::default();
    let gaussian = Gaussian::new(0.0, 1.0);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Independent(&uniform, &mut source).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| gaussian.inv_cdf(p)).collect::<Vec<_>>()));
}

#[bench]
fn sample(bencher: &mut Bencher) {
    let mut source = random::XorshiftPlus::new([42, 42]);
    let gaussian = Gaussian::new(0.0, 1.0);

    bencher.iter(|| black_box(gaussian.sample(&mut source)));
}
