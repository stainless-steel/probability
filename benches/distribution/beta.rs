use probability::prelude::*;
use test;

#[bench]
fn cdf(bench: &mut test::Bencher) {
    let beta = Beta::new(0.5, 1.5, 0.0, 1.0);
    let x = Sampler(&beta, &mut generator()).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(x.iter().map(|&x| beta.cdf(x)).collect::<Vec<_>>())
    });
}

#[bench]
fn inv_cdf(bench: &mut test::Bencher) {
    let beta = Beta::new(0.5, 1.5, 0.0, 1.0);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Sampler(&uniform, &mut generator()).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(p.iter().map(|&p| beta.inv_cdf(p)).collect::<Vec<_>>())
    });
}
