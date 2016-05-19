use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let d = Categorical::new(&[0.1; 10]);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source::default()).take(1000)
                                                                        .collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>()));
}
