use std::fmt;
use std::iter::IntoIterator;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum Bit {
    High,
    Low,
}

impl Bit {
    pub fn new(value: u8) -> Self {
        match value {
            0 => Bit::Low,
            1 => Bit::High,
            _ => panic!("Expected value of 1 or 0"),
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
            gate_input_2: Some(input_2.clone()),
            gate_output: Self::evaluate_and(input_1, input_2),
        }
    }

    pub fn new_or(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Or,
            gate_input_1: input_1,
            gate_input_2: Some(input_2.clone()),
            gate_output: Self::evaluate_or(input_1, input_2),
        }
    }

    pub fn new_xor(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Xor,
            gate_input_1: input_1,
            gate_input_2: Some(input_2.clone()),
            gate_output: Self::evaluate_xor(input_1, input_2),
        }
    }

    pub fn new_nor(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Nor,
            gate_input_1: input_1,
            gate_input_2: Some(input_2.clone()),
            gate_output: Self::evaluate_nor(input_1, input_2),
        }
    }

    pub fn new_nand(input_1: Bit, input_2: Bit) -> Self {
        Self {
            gate: GateType::Nand,
            gate_input_1: input_1,
            gate_input_2: Some(input_2.clone()),
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
pub struct Number {
    size: u8,
    bits: Vec<Bit>,
}

impl Number {
    pub fn new_with_size(n: u128, size: u8) -> Self {
        Self {
            size,
            bits: Self::create_binary_representation(n, size),
        }
    }

    pub fn new(n: u128) -> Self {
        let size = Self::calculate_size(n);
        Self {
            size,
            bits: Self::create_binary_representation(n, size),
        }
    }

    fn create_binary_representation(n: u128, size: u8) -> Vec<Bit> {
        let mut n_mut = n.clone();
        let mut binary = Vec::new();
        for i in (0..size).rev() {
            if 2u128.pow(i.into()) <= n_mut {
                n_mut -= 2u128.pow(i.into());
                binary.push(Bit::High);
            } else {
                binary.push(Bit::Low);
            }
        }
        binary
    }

    fn calculate_size(n: u128) -> u8 {
        (n as f64).log2().ceil() as u8
    }
}

impl IntoIterator for Number {
    type Item = Bit;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.bits.into_iter()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Adder {
    n1: Number,
    n2: Number,
}

impl Adder {
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

    fn adder_1bit(a: Bit, b: Bit, carry_in: Bit) -> (Bit, Bit) {
        let xor_1 = Gate::new_xor(a, b);
        let xor_2 = Gate::new_xor(xor_1.get_output(), carry_in);
        let and_1 = Gate::new_and(a, b);
        let and_2 = Gate::new_and(xor_1.get_output(), carry_in);
        let or = Gate::new_or(and_1.get_output(), and_2.get_output());
        (xor_2.get_output(), or.get_output())
    }

    fn adder_nbit(n1: Number, n2: Number) -> Number {
        let mut result: Vec<Bit> = Vec::new();
        let mut last_carry = Bit::Low;
        for (index, (a, b)) in n1
            .clone()
            .into_iter()
            .rev()
            .zip(n2.clone().into_iter().rev())
            .enumerate()
        {
            let (res, carry) = Self::adder_1bit(a, b, last_carry);
            last_carry = carry;
            result.push(res);
            if let Bit::High = last_carry {
                if index == (n1.size - 1).into() {
                    result.push(last_carry);
                }
            }
        }
        result.reverse();
        Number {
            size: result.len() as u8,
            bits: result,
        }
    }

    pub fn get_result(&self) -> Number {
        Self::adder_nbit(self.n1.clone(), self.n2.clone())
    }
}

#[cfg(test)]
mod test {
    use super::{Adder, Bit::*, Number};
    #[test]
    fn test_number_init_with_explicit_size() {
        let n = Number::new_with_size(42u128, 8);
        assert_eq!(
            n,
            Number {
                size: 8,
                bits: vec![Low, Low, High, Low, High, Low, High, Low]
            }
        );
    }

    #[test]
    fn test_number_init_with_implicit_size() {
        let n = Number::new(42);
        assert_eq!(
            n,
            Number {
                size: 6,
                bits: vec![High, Low, High, Low, High, Low],
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
                    size: 7,
                    bits: vec![High, Low, Low, Low, High, Low, High],
                },
                n2: Number {
                    size: 7,
                    bits: vec![Low, High, Low, High, Low, High, Low],
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
                size: 7,
                bits: vec![High, High, Low, High, High, High, High]
            }
        )
    }
}
