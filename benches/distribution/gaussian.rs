use probability::prelude::*;
use test;

#[bench]
fn cdf(bench: &mut test::Bencher) {
    let mut generator = generator();
    let gaussian = Gaussian::new(0.0, 1.0);
    let x = Sampler(&gaussian, &mut generator).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(x.iter().map(|&x| gaussian.cdf(x)).collect::<Vec<_>>())
    })
}

#[bench]
fn inv_cdf(bench: &mut test::Bencher) {
    let mut generator = generator();
    let gaussian = Gaussian::new(0.0, 1.0);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Sampler(&uniform, &mut generator).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(p.iter().map(|&p| gaussian.inv_cdf(p)).collect::<Vec<_>>())
    })
}
