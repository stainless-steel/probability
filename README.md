# Probability [![Version][version-img]][version-url] [![Status][status-img]][status-url]

The package provides a probability-theory toolbox.

## [Documentation][docs]

## Example

```rust
use probability::prelude::*;

let mut source = random::default();
let uniform = Uniform::new(0.0, 1.0);
let samples = Independent(&uniform, &mut source).take(10).collect::<Vec<_>>();
```

## Contributing

1. Fork the project.
2. Implement your idea.
3. Open a pull request.

[version-img]: http://stainless-steel.github.io/images/crates.svg
[version-url]: https://crates.io/crates/probability
[status-img]: https://travis-ci.org/stainless-steel/probability.svg?branch=master
[status-url]: https://travis-ci.org/stainless-steel/probability
[docs]: https://stainless-steel.github.io/probability
