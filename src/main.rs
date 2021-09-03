mod circuit;
use circuit::*;

fn main() {
    let a = Adder::new(42, 69);
    println!("{:?}", a.get_result());
}
