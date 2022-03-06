use probability::prelude::*;
use test::{black_box, Bencher};

#[bench]
fn distribution(bencher: &mut Bencher) {
    let d = Cauchy::new(0.0, 1.0);
    let x = Independent(&d, &mut source::default())
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>()));
}

#[bench]
fn inverse(bencher: &mut Bencher) {
    let d = Cauchy::new(0.0, 1.0);
    let p = Independent(&Uniform::new(0.0, 1.0), &mut source::default())
        .take(1000)
        .collect::<Vec<_>>();

    bencher.iter(|| black_box(p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>()));
}

#[bench]
fn sample(bencher: &mut Bencher) {
    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Cauchy::new(0.0, 1.0);

    bencher.iter(|| black_box(d.sample(&mut source)));
}

#[bench]
fn sample_directly(bencher: &mut Bencher) {
    fn sample(d: &Cauchy, source: &mut source::Xorshift128Plus) -> f64 {
        d.inverse(source::Source::read::<f64>(source))
    }

    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Cauchy::new(0.0, 1.0);

    bencher.iter(|| black_box(sample(&d, &mut source)));
}

#[bench]
fn sample_indirectly(bencher: &mut Bencher) {
    fn sample(d: &Cauchy, source: &mut source::Xorshift128Plus) -> f64 {
        let gaussian = Gaussian::new(0.0, 1.0);
        let a = gaussian.sample(source);
        let b = gaussian.sample(source);
        d.x_0() + d.gamma() * a / (b.abs() + f64::MIN_POSITIVE)
    }

    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Cauchy::new(0.0, 1.0);

    bencher.iter(|| black_box(sample(&d, &mut source)));
}
