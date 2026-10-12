//! The Poseidon2 sponge, with the instance the specification pins.
//!
//! The difference from Poseidon is the linear layer and the thinner internal
//! round, and both are visible here: `external` and `internal` are a sum and
//! `t` multiplications apiece, where Poseidon's matrix multiply is `t^2`.
use ark_bn254::Fr;
use ark_ff::{BigInt, Field, PrimeField, Zero};
use sha2::{Digest, Sha256};

use crate::fherma::{Inputs, Outputs, Point, Tensor};

const ALPHA: u64 = 5;
const EXTERNAL: usize = 8;
const INTERNAL: usize = 56;
const DOMAIN: &[u8] = b"fherma/poseidon2/bn254/x5";

/// The diagonal added to the all-ones matrix for the internal rounds.
const MU: [u64; 3] = [1, 2, 3];

fn stream(tag: &[u8], count: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(count);
    let mut counter: u32 = 0;
    while out.len() < count {
        let mut hasher = Sha256::new();
        hasher.update(tag);
        hasher.update(counter.to_be_bytes());
        let digest = hasher.finalize();
        counter += 1;
        let mut words = [0u64; 4];
        for (i, chunk) in digest.chunks_exact(8).rev().enumerate() {
            words[i] = u64::from_be_bytes(chunk.try_into().unwrap());
        }
        if let Some(value) = Fr::from_bigint(BigInt::new(words)) {
            out.push(value);
        }
    }
    out
}

pub struct State {
    t: usize,
    limbs: usize,
    ext: Vec<Fr>,
    intr: Vec<Fr>,
    mu: Vec<Fr>,
}

impl Default for State {
    fn default() -> Self {
        State { t: 0, limbs: 0, ext: Vec::new(), intr: Vec::new(), mu: Vec::new() }
    }
}

pub fn init(p: &Point) -> State {
    let t = p.T as usize;
    let mut tag = DOMAIN.to_vec();
    tag.extend_from_slice(b"/ark/");
    tag.extend_from_slice(t.to_string().as_bytes());
    let both = stream(&tag, EXTERNAL * t + INTERNAL);
    let (ext, intr) = both.split_at(EXTERNAL * t);
    State {
        t,
        limbs: p.L as usize,
        ext: ext.to_vec(),
        intr: intr.to_vec(),
        mu: MU[..t].iter().map(|&m| Fr::from(m)).collect(),
    }
}

/// M_E: 2 on the diagonal, 1 off it — a sum, then one addition per element.
fn external(s: &mut [Fr]) {
    let total: Fr = s.iter().sum();
    for x in s.iter_mut() {
        *x += total;
    }
}

/// M_I: the all-ones matrix plus diag(MU).
fn internal(s: &mut [Fr], mu: &[Fr]) {
    let total: Fr = s.iter().sum();
    for (x, m) in s.iter_mut().zip(mu) {
        *x = total + *m * *x;
    }
}

fn permute(s: &mut [Fr], state: &State) {
    let t = state.t;
    let half = EXTERNAL / 2;
    external(s);
    for rnd in 0..half {
        let base = rnd * t;
        for i in 0..t {
            s[i] = (s[i] + state.ext[base + i]).pow([ALPHA]);
        }
        external(s);
    }
    for rnd in 0..INTERNAL {
        s[0] = (s[0] + state.intr[rnd]).pow([ALPHA]);
        internal(s, &state.mu);
    }
    for rnd in half..EXTERNAL {
        let base = rnd * t;
        for i in 0..t {
            s[i] = (s[i] + state.ext[base + i]).pow([ALPHA]);
        }
        external(s);
    }
}

pub fn run(state: &mut State, inp: &Inputs) -> Outputs {
    let (t, limbs) = (state.t, state.limbs);
    let n = inp.input.data.len() / limbs;
    let rate = t - 1;

    let mut s = vec![Fr::zero(); t];
    s[t - 1] = Fr::from(n as u64);

    let mut at = 0;
    loop {
        for j in 0..rate {
            if at + j < n {
                let mut words = [0u64; 4];
                words[..limbs]
                    .copy_from_slice(&inp.input.data[(at + j) * limbs..(at + j + 1) * limbs]);
                s[j] += Fr::from_bigint(BigInt::new(words)).expect("below the modulus");
            }
        }
        permute(&mut s, state);
        at += rate;
        if at >= n {
            break;
        }
    }

    Outputs {
        digest: Tensor { shape: vec![limbs as i64], data: s[0].into_bigint().0[..limbs].to_vec() },
    }
}

pub fn free(state: State) {
    drop(state);
}
