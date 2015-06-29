# Probability [![Version][version-img]][version-url] [![Status][status-img]][status-url]

The package provides a probability-theory toolbox.

## [Documentation][docs]

## Example

```rust
use probability::prelude::*;

let uniform = Uniform::new(0.0, 1.0);
let mut generator = generator::default();
let samples = Independent(&uniform, &mut generator).take(10).collect::<Vec<_>>();
```

## Contributing

1. Fork the project.
2. Implement your idea.
3. Open a pull request.

[version-img]: https://img.shields.io/crates/v/probability.svg
[version-url]: https://crates.io/crates/probability
[status-img]: https://travis-ci.org/stainless-steel/probability.svg?branch=master
[status-url]: https://travis-ci.org/stainless-steel/probability
[docs]: https://stainless-steel.github.io/probability
