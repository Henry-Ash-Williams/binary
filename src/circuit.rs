#![allow(dead_code)]

use std::fmt;
use std::iter::{FromIterator, IntoIterator};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum Bit {
    High,
    Low,
}

impl Bit {
    pub fn new(value: bool) -> Self {
        match value {
            true => Bit::High,
            _ => Bit::Low,
        }
    }

    pub fn flip(&self) -> Self {
        match *self {
            Bit::Low => Bit::High,
            Bit::High => Bit::Low,
        }
    }
}

impl fmt::Display for Bit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Bit::High => "1",
                Bit::Low => "0",
            }
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum GateType {
    Not,
    Or,
    And,
    Xor,
    Nand,
    Nor,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct Gate {
    gate: GateType,
    gate_input_1: Bit,
    gate_input_2: Option<Bit>,
    gate_output: Bit,
}

impl Gate {
    pub fn get_output(&self) -> Bit {
        self.gate_output
    }

    pub fn new_not(input_1: Bit) -> Self {
        Self {
            gate: GateType::Not,
            gate_input_1: input_1,
            gate_input_2: None,
            gate_output: Self::evaluate_not(input_1),
        }
    }

    pub fn new_and(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::And,
            gate_input_1: input_1,
            gate_input_2: Some(input_2),
            gate_output: Self::evaluate_and(input_1, input_2),
        }
    }

    pub fn new_or(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Or,
            gate_input_1: input_1,
            gate_input_2: Some(input_2),
            gate_output: Self::evaluate_or(input_1, input_2),
        }
    }

    pub fn new_xor(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Xor,
            gate_input_1: input_1,
            gate_input_2: Some(input_2),
            gate_output: Self::evaluate_xor(input_1, input_2),
        }
    }

    pub fn new_nor(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Nor,
            gate_input_1: input_1,
            gate_input_2: Some(input_2),
            gate_output: Self::evaluate_nor(input_1, input_2),
        }
    }

    pub fn new_nand(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Nand,
            gate_input_1: input_1,
            gate_input_2: Some(input_2),
            gate_output: Self::evaluate_nand(input_1, input_2),
        }
    }

    fn evaluate_not(input_1: Bit) -> Bit {
        use Bit::*;
        match input_1 {
            Low => High,
            High => Low,
        }
    }

    fn evaluate_or(input_1: Bit, input_2: Bit) -> Bit {
        use Bit::*;
        match (input_1, input_2) {
            (Low, Low) => Low,
            _ => High,
        }
    }

    fn evaluate_and(input_1: Bit, input_2: Bit) -> Bit {
        use Bit::*;
        match (input_1, input_2) {
            (High, High) => High,
            _ => Low,
        }
    }

    fn evaluate_xor(input_1: Bit, input_2: Bit) -> Bit {
        use Bit::*;
        match (input_1, input_2) {
            (Low, Low) | (High, High) => Low,
            _ => High,
        }
    }

    fn evaluate_nor(input_1: Bit, input_2: Bit) -> Bit {
        Gate::evaluate_not(Gate::evaluate_or(input_1, input_2))
    }

    fn evaluate_nand(input_1: Bit, input_2: Bit) -> Bit {
        Gate::evaluate_not(Gate::evaluate_and(input_1, input_2))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Number(Vec<Bit>);

impl Number {
    /// Initialize a binary number representing zero
    pub fn zero() -> Self {
        Self { 0: vec![] }
    }

    /// Initialize a number with an explicit size, if size > the number of bits required
    /// to store the number, remaining bits will store `Bit::Low`
    pub fn new_with_size(n: u128, size: u8) -> Self {
        Self {
            0: Self::create_binary_representation(n, size),
        }
    }

    /// Initialize a number with an implicit size, leftmost bit will always be `Bit::High`
    pub fn new(n: u128) -> Self {
        Self {
            0: Self::create_binary_representation(n, Self::calculate_size(n)),
        }
    }

    /// Creates the binary representation of the number
    fn create_binary_representation(n: u128, size: u8) -> Vec<Bit> {
        (0..size)
            .rev()
            .map(|i| 1 << i)
            .map(|i| n & i != 0)
            .map(|b| if b { Bit::High } else { Bit::Low })
            .collect::<Vec<_>>()
    }
    
    /// Returns the number of bits necessary to store the number 
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Calculates the number of bits required to store the number
    fn calculate_size(n: u128) -> u8 {
        (n as f64).log2().ceil() as u8
    }

    /// Appends a bit to the number
    pub fn push(&mut self, bit: Bit) {
        self.0.push(bit);
    }

    pub fn pad_number(&self, size: usize) -> Number {
        let mut number_clone = self.0.clone();
        number_clone.reverse(); 
        for i in 0..self.size() - size {
            number_clone.push(Bit::Low); 
        }
        number_clone.reverse(); 
        Number { 0: number_clone }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::new();

        for i in self.clone().into_iter() {
            output.push_str(&format!("{}", i));
        }

        write!(f, "{}", output)
    }
}

impl IntoIterator for Number {
    type Item = Bit;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<Bit> for Number {
    fn from_iter<I: IntoIterator<Item = Bit>>(iter: I) -> Self {
        let mut zero = Number::zero();

        for i in iter {
            zero.push(i);
        }

        zero
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Adder {
    n1: Number,
    n2: Number,
}

impl Adder {
    /// Creates a new addition circuit
    pub fn new(n1: u128, n2: u128) -> Self {
        let nbits = std::cmp::max(
            (n1 as f64).log2().ceil() as u8,
            (n2 as f64).log2().ceil() as u8,
        );

        Self {
            n1: Number::new_with_size(n1, nbits),
            n2: Number::new_with_size(n2, nbits),
        }
    }
    
    /// Creates new addition circuit using numbers as inputs 
    pub fn new_from_number(n1: Number, n2: Number) -> Self {
        let size = std::cmp::max(
            n1.size(),
            n2.size()
        ); 
        Self { n1: n1.pad_number(size), n2: n2.pad_number(size) }
    }
    
    /// 1-bit addition circuit 
    fn adder_1bit(a: Bit, b: Bit, carry_in: Bit) -> (Bit, Bit) {
        let xor_1 = Gate::new_xor(a, b);
        let xor_2 = Gate::new_xor(xor_1.get_output(), carry_in);
        let and_1 = Gate::new_and(a, b);
        let and_2 = Gate::new_and(xor_1.get_output(), carry_in);
        let or = Gate::new_or(and_1.get_output(), and_2.get_output());
        (xor_2.get_output(), or.get_output())
    }
    
    /// n-bit addition circuit 
    fn adder_nbit(n1: Number, n2: Number) -> Number {
        let mut result: Vec<Bit> = Vec::new();
        let mut last_carry = Bit::Low;
        for (index, (a, b)) in n1
            .clone()
            .into_iter()
            .rev()
            .zip(n2.into_iter().rev())
            .enumerate()
        {
            let (res, carry) = Self::adder_1bit(a, b, last_carry);
            last_carry = carry;
            result.push(res);
            if let Bit::High = last_carry {
                if index == n1.size() - 1 {
                    result.push(last_carry);
                }
            }
        }
        result.reverse();
        Number { 0: result }
    }
    
    /// Calculates and returns the number returned by the addition circuit 
    pub fn get_result(&self) -> Number {
        Self::adder_nbit(self.n1.clone(), self.n2.clone())
    }
}

#[cfg(test)]
mod test {
    use super::{Adder, Bit::*, Gate, Number};

    #[test]
    fn test_not() {
        let not1 = Gate::new_not(High);
        let not2 = Gate::new_not(Low);

        assert_eq!(not1.get_output(), Low);
        assert_eq!(not2.get_output(), High);
    }

    #[test]
    fn test_or() {
        let or1 = Gate::new_or(Low, Low);
        let or2 = Gate::new_or(High, Low);
        let or3 = Gate::new_or(Low, High);
        let or4 = Gate::new_or(High, High);

        assert_eq!(or1.get_output(), Low);
        assert_eq!(or2.get_output(), High);
        assert_eq!(or3.get_output(), High);
        assert_eq!(or4.get_output(), High);
    }

    #[test]
    fn test_and() {
        let and1 = Gate::new_and(Low, Low);
        let and2 = Gate::new_and(High, Low);
        let and3 = Gate::new_and(Low, High);
        let and4 = Gate::new_and(High, High);

        assert_eq!(and1.get_output(), Low);
        assert_eq!(and2.get_output(), Low);
        assert_eq!(and3.get_output(), Low);
        assert_eq!(and4.get_output(), High);
    }

    #[test]
    fn test_xor() {
        let xor1 = Gate::new_xor(Low, Low);
        let xor2 = Gate::new_xor(High, Low);
        let xor3 = Gate::new_xor(Low, High);
        let xor4 = Gate::new_xor(High, High);

        assert_eq!(xor1.get_output(), Low);
        assert_eq!(xor2.get_output(), High);
        assert_eq!(xor3.get_output(), High);
        assert_eq!(xor4.get_output(), Low);
    }

    #[test]
    fn test_nor() {
        let nor1 = Gate::new_nor(Low, Low);
        let nor2 = Gate::new_nor(High, Low);
        let nor3 = Gate::new_nor(Low, High);
        let nor4 = Gate::new_nor(High, High);

        assert_eq!(nor1.get_output(), High);
        assert_eq!(nor2.get_output(), Low);
        assert_eq!(nor3.get_output(), Low);
        assert_eq!(nor4.get_output(), Low);
    }

    #[test]
    fn test_nand() {
        let nand1 = Gate::new_nand(Low, Low);
        let nand2 = Gate::new_nand(High, Low);
        let nand3 = Gate::new_nand(Low, High);
        let nand4 = Gate::new_nand(High, High);

        assert_eq!(nand1.get_output(), High);
        assert_eq!(nand2.get_output(), High);
        assert_eq!(nand3.get_output(), High);
        assert_eq!(nand4.get_output(), Low);
    }

    #[test]
    fn test_number_init_with_explicit_size() {
        let n = Number::new_with_size(42u128, 8);
        assert_eq!(
            n,
            Number {
                0: vec![Low, Low, High, Low, High, Low, High, Low]
            }
        );
    }

    #[test]
    fn test_number_init_with_implicit_size() {
        let n = Number::new(42);
        assert_eq!(
            n,
            Number {
                0: vec![High, Low, High, Low, High, Low],
            }
        );
    }

    #[test]
    fn test_number_into_iter() {
        let n = Number::new(42);

        let mut n_iter = n.into_iter();

        assert_eq!(n_iter.next(), Some(High));
        assert_eq!(n_iter.next(), Some(Low));
        assert_eq!(n_iter.next(), Some(High));
    }

    #[test]
    fn test_adder_init() {
        let a = Adder::new(69, 42);

        assert_eq!(
            a,
            Adder {
                n1: Number {
                    0: vec![High, Low, Low, Low, High, Low, High],
                },
                n2: Number {
                    0: vec![Low, High, Low, High, Low, High, Low],
                },
            },
        )
    }
    #[test]
    fn test_adder_addition() {
        let a = Adder::new(42, 69);

        assert_eq!(
            a.get_result(),
            Number {
                0: vec![High, High, Low, High, High, High, High]
            }
        )
    }
}
