use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let categorical = Categorical::new(&[0.1; 10]);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Independent(&uniform, &mut random::default()).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| categorical.inv_cdf(p)).collect::<Vec<_>>()));
}
