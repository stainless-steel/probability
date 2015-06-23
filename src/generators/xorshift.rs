use Generator;

/// The Xorshift+ algorithm.
///
/// ## References
///
/// 1. Sebastiano Vigna, “Further Scramblings of Marsaglia’s Xorshift
///    Generators,” CoRR, 2014.
///
/// 2. https://en.wikipedia.org/wiki/Xorshift
pub struct XorshiftPlus {
    state: [u64; 2],
}

impl XorshiftPlus {
    /// Create a generator.
    #[inline(always)]
    pub fn new(seed: [u64; 2]) -> XorshiftPlus {
        XorshiftPlus { state: seed }
    }
}

impl Generator for XorshiftPlus {
    #[inline(always)]
    fn read(&mut self) -> u64 {
        let (mut x, y) = (self.state[0], self.state[1]);

        self.state[0] = y;
        x = x ^ (x << 23);
        x = x ^ (x >> 17);
        x = x ^ y ^ (y >> 26);
        self.state[1] = x;

        x.wrapping_add(y)
    }
}
