use probability::prelude::*;
use test::{black_box, Bencher};

#[bench]
fn distribution(bencher: &mut Bencher) {
    let d = Binomial::new(100_000, 0.845);
    let x = Independent(&d, &mut source::default())
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| {
        black_box(
            x.iter()
                .map(|&x| d.distribution(x as f64))
                .collect::<Vec<_>>(),
        )
    });
}

#[bench]
fn inverse(bencher: &mut Bencher) {
    let d = Binomial::new(100_000, 0.845);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source::default())
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>()));
}

#[bench]
fn mass(bencher: &mut Bencher) {
    let d = Binomial::new(100_000, 0.845);
    let x = Independent(&d, &mut source::default())
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| d.mass(x)).collect::<Vec<_>>()));
}
