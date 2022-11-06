use probability::prelude::*;
use test::{black_box, Bencher};

#[bench]
fn inverse(bencher: &mut Bencher) {
    let d = Categorical::new(&[0.1; 10]);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source::default([42, 60]))
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>()));
}
