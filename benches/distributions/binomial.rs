use probability::{Distribution, Sampler, generator};
use probability::distributions::{Binomial, Uniform};
use test;

#[bench]
fn cdf(bench: &mut test::Bencher) {
    let binom = Binomial::new(100_000, 0.845);
    let x = Sampler(&binom, &mut generator()).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(x.iter().map(|&x| binom.cdf(x)).collect::<Vec<_>>())
    });
}

#[bench]
fn inv_cdf(bench: &mut test::Bencher) {
    let binom = Binomial::new(100_000, 0.845);
    let uniform = Uniform::new(0.0, 1.0);
    let p = Sampler(&uniform, &mut generator()).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(p.iter().map(|&p| binom.inv_cdf(p)).collect::<Vec<_>>())
    });
}

#[bench]
fn pdf(bench: &mut test::Bencher) {
    let binom = Binomial::new(100_000, 0.845);
    let x = Sampler(&binom, &mut generator()).take(1000).collect::<Vec<_>>();

    bench.iter(|| {
        test::black_box(x.iter().map(|&x| binom.pdf(x)).collect::<Vec<_>>())
    });
}
