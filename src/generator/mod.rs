//! Sources of randomness.

use std::cell::RefCell;
use std::rc::Rc;

/// A source of randomness.
pub trait Generator {
    /// Read the next chunk.
    fn read(&mut self) -> u64;

    /// Read the next quantity.
    #[inline(always)]
    fn next<T: Quantity>(&mut self) -> T {
        Quantity::make(self.read())
    }
}

/// A random quantity.
pub trait Quantity {
    /// Make up a random quantity.
    fn make(u64) -> Self;
}

impl Quantity for f64 {
    #[inline(always)]
    fn make(chunk: u64) -> f64 {
        chunk as f64 / (::std::u64::MAX as f64 + 1.0)
    }
}

impl Quantity for u64 {
    #[inline(always)]
    fn make(chunk: u64) -> u64 {
        chunk
    }
}

/// The default generator, which is the Xorshift+ algorithm.
pub struct Default(Rc<RefCell<XorshiftPlus>>);

impl Default {
    /// Seed the generator.
    #[inline(always)]
    pub fn seed(&mut self, seed: [u64; 2]) -> &mut Default {
        *self.0.borrow_mut() = XorshiftPlus::new(seed);
        self
    }
}

impl Generator for Default {
    #[inline(always)]
    fn read(&mut self) -> u64 {
        self.0.borrow_mut().read()
    }

    #[inline(always)]
    fn next<T: Quantity>(&mut self) -> T {
        self.0.borrow_mut().next()
    }
}

/// Return the default generator.
///
/// Each thread has its own copy of the generator, and each copy is initialized
/// with the same default seed. Consequently, the usage is thread safe; however,
/// each thread is responsible for reseeding its default generator.
#[inline(always)]
pub fn default() -> Default {
    thread_local!(static DEFAULT: Rc<RefCell<XorshiftPlus>> = {
        Rc::new(RefCell::new(XorshiftPlus::new([42, 69])))
    });
    Default(DEFAULT.with(|generator| generator.clone()))
}

mod xorshift;

pub use self::xorshift::XorshiftPlus;
