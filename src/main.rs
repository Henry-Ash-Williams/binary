mod circuit;
use circuit::*;

const NUMBER_1: u32 = 12943; 
const NUMBER_2: u32 = 53211;

fn main() {
    let n1 = Number::new(NUMBER_1.into()); 
    let n2 = Number::new(NUMBER_2.into()); 
    let adder = Adder::new_from_number(n1.clone(), n2.clone()); 
    
    println!("  {}", n1); 
    println!("+ {}", n2); 
    println!("=========="); 
    println!("  {}", adder.get_result());
}
