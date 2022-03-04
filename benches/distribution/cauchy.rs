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

/// Benchmark sampling via the implemented method.
#[bench]
fn sample(bencher: &mut Bencher) {
    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Cauchy::new(0.0, 1.0);

    bencher.iter(|| black_box(d.sample(&mut source)));
}

/// Benchmark sampling via the direct method (applying the quantile function to
/// a uniformly distributed random variable), regardless of how sampling is
/// actually implemented for `Cauchy`.
#[bench]
fn sample_directly(bencher: &mut Bencher) {
    fn draw_sample(d: &Cauchy, source: &mut source::Xorshift128Plus) -> f64 {
        d.inverse(source::Source::read::<f64>(source))
    }

    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Cauchy::new(0.0, 1.0);

    bencher.iter(|| black_box(draw_sample(&d, &mut source)));
}

/// Benchmark sampling via the indirect method (taking the ratio of two
/// standard normal distributed random variables), regardless of how sampling
/// is actually implemented for `Cauchy`.
#[bench]
fn sample_indirectly(bencher: &mut Bencher) {
    fn draw_sample(d: &Cauchy, source: &mut source::Xorshift128Plus) -> f64 {
        let gaussian = Gaussian::new(0.0, 1.0);
        let a = gaussian.sample(source);
        let b = gaussian.sample(source);
        d.loc() + d.gamma() * a / (b.abs() + f64::EPSILON)
    }

    let mut source = source::Xorshift128Plus::new([42, 69]);
    let d = Cauchy::new(0.0, 1.0);

    bencher.iter(|| black_box(draw_sample(&d, &mut source)));
}
