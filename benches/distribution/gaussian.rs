use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn cdf(bencher: &mut Bencher) {
    let mut source = source::default();
    let d = Gaussian::new(0.0, 1.0);
    let x = Independent(&d, &mut source).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| d.cdf(x)).collect::<Vec<_>>()));
}

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let mut source = source::default();
    let d = Gaussian::new(0.0, 1.0);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>()));
}

#[bench]
fn sample(bencher: &mut Bencher) {
    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Gaussian::new(0.0, 1.0);

    bencher.iter(|| black_box(d.sample(&mut source)));
}
