# Probability [![Version][version-img]][version-url] [![Status][status-img]][status-url]

The package provides a probability-theory toolbox.

## [Documentation][doc]

## Example

```rust
use probability::prelude::*;

let mut source = source::default();
let distribution = Uniform::new(0.0, 1.0);
let mut sampler = Independent(&distribution, &mut source);
let samples = sampler.take(10).collect::<Vec<_>>();
```

## Contribution

Your contribution is highly appreciated. Do not hesitate to open an issue or a
pull request. Note that any contribution submitted for inclusion in the project
will be licensed according to the terms given in [LICENSE.md](LICENSE.md).

[doc]: https://stainless-steel.github.io/probability
[status-img]: https://travis-ci.org/stainless-steel/probability.svg?branch=master
[status-url]: https://travis-ci.org/stainless-steel/probability
[version-img]: https://img.shields.io/crates/v/probability.svg
[version-url]: https://crates.io/crates/probability
