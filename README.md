# Probability [![Package][package-img]][package-url] [![Documentation][documentation-img]][documentation-url] [![Build][build-img]][build-url]

The package provides a probability-theory toolbox.

## Example

```rust
use probability::prelude::*;

let mut source = source::default();
let distribution = Uniform::new(0.0, 1.0);
let sampler = Independent(&distribution, &mut source);
let samples = sampler.take(10).collect::<Vec<_>>();
```

Sources of randomness are provided by the [`random`][random] crate via the
`source` module. In addition, one can make use of those sources that are
available in the [`rand`][rand] crate as illustrated below:

```rust
use probability::prelude::*;

struct Source<T>(T);

impl<T: rand::RngCore> source::Source for Source<T> {
    fn read_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
}

let mut source = Source(rand::rngs::OsRng::new().unwrap());
let distribution = Uniform::new(0.0, 1.0);
let sampler = Independent(&distribution, &mut source);
let samples = sampler.take(10).collect::<Vec<_>>();
```

## Contribution

Your contribution is highly appreciated. Do not hesitate to open an issue or a
pull request. Note that any contribution submitted for inclusion in the project
will be licensed according to the terms given in [LICENSE.md](LICENSE.md).

[rand]: https://crates.io/crates/rand
[random]: https://crates.io/crates/random

[build-img]: https://travis-ci.org/stainless-steel/probability.svg?branch=master
[build-url]: https://travis-ci.org/stainless-steel/probability
[documentation-img]: https://docs.rs/probability/badge.svg
[documentation-url]: https://docs.rs/probability
[package-img]: https://img.shields.io/crates/v/probability.svg
[package-url]: https://crates.io/crates/probability
