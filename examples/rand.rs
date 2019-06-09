extern crate probability;
extern crate rand;

fn main() {
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
    print!("{:?}", samples);
}
