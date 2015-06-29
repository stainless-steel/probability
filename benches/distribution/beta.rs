use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn cdf(bencher: &mut Bencher) {
    let beta = Beta::new(0.5, 1.5, 0.0, 1.0);
    let x = Independent(&beta, &mut generator::default()).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| beta.cdf(x)).collect::<Vec<_>>()));
}

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let beta = Beta::new(0.5, 1.5, 0.0, 1.0);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Independent(&uniform, &mut generator::default()).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| beta.inv_cdf(p)).collect::<Vec<_>>()));
}
