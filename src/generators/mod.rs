//! Sources of randomness.

use {Generator, Quantity};
use std::cell::RefCell;
use std::rc::Rc;

mod xorshift;

pub use self::xorshift::XorshiftPlus;

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
#[inline(always)]
pub fn default() -> Default {
    thread_local!(static DEFAULT: Rc<RefCell<XorshiftPlus>> = {
        Rc::new(RefCell::new(XorshiftPlus::new([42, 69])))
    });
    Default(DEFAULT.with(|generator| generator.clone()))
}
