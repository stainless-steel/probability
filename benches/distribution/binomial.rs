use probability::prelude::*;
use test::{Bencher, black_box};

#[bench]
fn cdf(bencher: &mut Bencher) {
    let d = Binomial::new(100_000, 0.845);
    let x = Independent(&d, &mut source::default()).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| d.cdf(x as f64)).collect::<Vec<_>>()));
}

#[bench]
fn pmf(bencher: &mut Bencher) {
    let d = Binomial::new(100_000, 0.845);
    let x = Independent(&d, &mut source::default()).take(1000).collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| d.pmf(x)).collect::<Vec<_>>()));
}

#[bench]
fn inv_cdf(bencher: &mut Bencher) {
    let d = Binomial::new(100_000, 0.845);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source::default()).take(1000)
                                                                        .collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>()));
}
