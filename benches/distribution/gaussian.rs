use probability::prelude::*;
use test::{black_box, Bencher};

#[bench]
fn distribution(bencher: &mut Bencher) {
    let d = Gaussian::new(0.0, 1.0);
    let x = Independent(&d, &mut source::default(42))
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>()));
}

#[bench]
fn inverse(bencher: &mut Bencher) {
    let d = Gaussian::new(0.0, 1.0);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source::default(42))
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>()));
}

#[bench]
fn sample(bencher: &mut Bencher) {
    let mut source = source::default(42);
    let d = Gaussian::new(0.0, 1.0);

    bencher.iter(|| black_box(d.sample(&mut source)));
}
