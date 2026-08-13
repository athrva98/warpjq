//! Aggregation state and the deterministic group ordering.
//!
//! Both backends funnel into these types: the CPU backend updates them
//! directly, and the GPU backend reduces on-device and then merges the
//! per-chunk partials here. Doing the final merge in one place means
//! `sum(.bytes)` cannot drift between backends just because the chunk count
//! changed.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::json::{Kind, Slot};
use crate::query::AggKind;

/// Running state for one aggregate over one group (or the whole stream).
#[derive(Clone, Copy, Debug)]
pub struct AggState {
    /// Lines that reached the aggregate, whether or not they had a value.
    pub count: u64,
    /// Lines that contributed a number.
    pub numeric: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    /// Set when a non-numeric, non-null value was seen where a number was
    /// expected. Surfaced once as a warning rather than per line.
    pub saw_non_numeric: bool,
}

impl Default for AggState {
    fn default() -> Self {
        AggState {
            count: 0,
            numeric: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            saw_non_numeric: false,
        }
    }
}

impl AggState {
    pub fn push_count(&mut self) {
        self.count += 1;
    }

    /// Feeds one extracted value. `null` and missing are skipped silently,
    /// matching how SQL aggregates treat NULL.
    pub fn push_value(&mut self, slot: &Slot<'_>) {
        self.count += 1;
        match slot.kind {
            Kind::Missing | Kind::Null => {}
            Kind::Num => match slot.as_f64() {
                Some(v) => self.push_number(v),
                None => self.saw_non_numeric = true,
            },
            _ => self.saw_non_numeric = true,
        }
    }

    pub fn push_number(&mut self, v: f64) {
        self.numeric += 1;
        self.sum += v;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }

    pub fn merge(&mut self, other: &AggState) {
        self.count += other.count;
        self.numeric += other.numeric;
        self.sum += other.sum;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.saw_non_numeric |= other.saw_non_numeric;
    }

    /// The final value for `kind`. `None` renders as JSON `null`, which is
    /// what min/max/avg of nothing should be.
    pub fn finish(&self, kind: AggKind) -> Option<f64> {
        match kind {
            AggKind::Count => Some(self.count as f64),
            // Summing nothing is 0, as in SQL and in jq's `add // 0` idiom.
            AggKind::Sum => Some(self.sum),
            AggKind::Min => (self.numeric > 0).then_some(self.min),
            AggKind::Max => (self.numeric > 0).then_some(self.max),
            AggKind::Avg => (self.numeric > 0).then(|| self.sum / self.numeric as f64),
        }
    }
}

/// A `group_by` key, owned so it can outlive the chunk it came from.
///
/// Ordering is jq's total order (null < false < true < numbers < strings),
/// which gives `group_by` a deterministic output sequence independent of hash
/// iteration order, chunk count, or backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GroupKey {
    Null,
    Bool(bool),
    /// Kept as raw bytes so the original spelling survives to the output, with
    /// the parsed value alongside for ordering.
    Num {
        raw: Vec<u8>,
        bits: u64,
    },
    Str(Vec<u8>),
    /// Arrays and objects used as group keys: compared by raw bytes.
    Composite(Vec<u8>),
}

impl GroupKey {
    pub fn from_slot(slot: &Slot<'_>) -> GroupKey {
        match slot.kind {
            Kind::Missing | Kind::Null => GroupKey::Null,
            Kind::Bool => GroupKey::Bool(slot.raw == b"true"),
            Kind::Num => {
                let v = slot.as_f64().unwrap_or(f64::NAN);
                GroupKey::Num {
                    raw: slot.raw.to_vec(),
                    bits: order_bits(v),
                }
            }
            Kind::Str => {
                let inner = slot.str_inner();
                if inner.contains(&b'\\') {
                    let mut buf = Vec::with_capacity(inner.len());
                    if crate::json::unescape_into(inner, &mut buf).is_ok() {
                        return GroupKey::Str(buf);
                    }
                }
                GroupKey::Str(inner.to_vec())
            }
            Kind::Arr | Kind::Obj => GroupKey::Composite(slot.raw.to_vec()),
        }
    }

    fn rank(&self) -> u8 {
        match self {
            GroupKey::Null => 0,
            GroupKey::Bool(false) => 1,
            GroupKey::Bool(true) => 2,
            GroupKey::Num { .. } => 3,
            GroupKey::Str(_) => 4,
            GroupKey::Composite(_) => 5,
        }
    }

    /// The key rendered as a JSON value, for the output row.
    pub fn to_json(&self) -> Vec<u8> {
        match self {
            GroupKey::Null => b"null".to_vec(),
            GroupKey::Bool(true) => b"true".to_vec(),
            GroupKey::Bool(false) => b"false".to_vec(),
            GroupKey::Num { raw, .. } => raw.clone(),
            GroupKey::Str(s) => {
                let mut out = Vec::with_capacity(s.len() + 2);
                crate::output::write_json_string(&mut out, s);
                out
            }
            GroupKey::Composite(raw) => raw.clone(),
        }
    }

    /// The key as plain text, for a CSV cell.
    pub fn to_plain(&self) -> Vec<u8> {
        match self {
            GroupKey::Str(s) => s.clone(),
            other => other.to_json(),
        }
    }
}

impl Ord for GroupKey {
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b) = (self.rank(), other.rank());
        if a != b {
            return a.cmp(&b);
        }
        match (self, other) {
            (GroupKey::Num { bits: x, .. }, GroupKey::Num { bits: y, .. }) => x.cmp(y),
            (GroupKey::Str(x), GroupKey::Str(y)) => x.cmp(y),
            (GroupKey::Composite(x), GroupKey::Composite(y)) => x.cmp(y),
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for GroupKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Maps a `f64` onto a `u64` whose unsigned order matches float order, so
/// group keys sort numerically without carrying a non-`Ord` float around.
fn order_bits(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

/// Accumulator for `group_by(...) | agg`.
#[derive(Default, Debug)]
pub struct GroupAccumulator {
    groups: HashMap<GroupKey, AggState>,
}

impl GroupAccumulator {
    pub fn entry(&mut self, key: GroupKey) -> &mut AggState {
        self.groups.entry(key).or_default()
    }

    pub fn merge(&mut self, other: GroupAccumulator) {
        for (k, v) in other.groups {
            self.groups.entry(k).or_default().merge(&v);
        }
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn saw_non_numeric(&self) -> bool {
        self.groups.values().any(|g| g.saw_non_numeric)
    }

    /// Groups in deterministic key order.
    pub fn sorted(self) -> Vec<(GroupKey, AggState)> {
        let mut v: Vec<_> = self.groups.into_iter().collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

/// Formats an aggregate result the way jq prints numbers: integral values
/// without a decimal point, everything else shortest-round-trip.
pub fn format_number(v: f64) -> String {
    if v.is_nan() {
        return "null".into();
    }
    if v.is_infinite() {
        // jq clamps infinities to the f64 extremes on output.
        return if v > 0.0 {
            "1.7976931348623157e+308".into()
        } else {
            "-1.7976931348623157e+308".into()
        };
    }
    // 2^53 is where consecutive integers stop being representable; past it a
    // "integer" rendering would be a lie, so fall through to scientific.
    if v.fract() == 0.0 && v.abs() < 9.007_199_254_740_992e15 {
        let mut buf = itoa::Buffer::new();
        return buf.format(v as i64).to_string();
    }
    let mut buf = ryu::Buffer::new();
    let s = buf.format(v);
    // ryu writes `1e300`; jq (and JSON tooling generally) writes `1e+300`.
    match s.split_once('e') {
        Some((mantissa, exp)) if !exp.starts_with('-') => format!("{mantissa}e+{exp}"),
        _ => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(raw: &str, kind: Kind) -> Slot<'_> {
        Slot {
            kind,
            raw: raw.as_bytes(),
        }
    }

    #[test]
    fn empty_aggregates_behave_like_sql() {
        let s = AggState::default();
        assert_eq!(s.finish(AggKind::Count), Some(0.0));
        assert_eq!(s.finish(AggKind::Sum), Some(0.0));
        assert_eq!(s.finish(AggKind::Min), None);
        assert_eq!(s.finish(AggKind::Max), None);
        assert_eq!(s.finish(AggKind::Avg), None);
    }

    #[test]
    fn nulls_are_skipped_by_numeric_aggregates_but_counted() {
        let mut s = AggState::default();
        s.push_value(&slot("10", Kind::Num));
        s.push_value(&slot("null", Kind::Null));
        s.push_value(&slot("20", Kind::Num));
        assert_eq!(s.finish(AggKind::Count), Some(3.0));
        assert_eq!(s.finish(AggKind::Sum), Some(30.0));
        assert_eq!(s.finish(AggKind::Avg), Some(15.0));
        assert_eq!(s.finish(AggKind::Min), Some(10.0));
    }

    #[test]
    fn merging_partials_matches_a_single_pass() {
        let vals = [3.0, 9.0, -2.0, 40.0, 0.5];
        let mut whole = AggState::default();
        for v in vals {
            whole.push_number(v);
        }
        let (mut a, mut b) = (AggState::default(), AggState::default());
        for v in &vals[..2] {
            a.push_number(*v);
        }
        for v in &vals[2..] {
            b.push_number(*v);
        }
        a.merge(&b);
        assert_eq!(a.sum, whole.sum);
        assert_eq!(a.min, whole.min);
        assert_eq!(a.max, whole.max);
        assert_eq!(a.numeric, whole.numeric);
    }

    #[test]
    fn group_keys_sort_in_jq_total_order() {
        let mut keys = vec![
            GroupKey::Str(b"b".to_vec()),
            GroupKey::Num {
                raw: b"10".to_vec(),
                bits: order_bits(10.0),
            },
            GroupKey::Null,
            GroupKey::Str(b"a".to_vec()),
            GroupKey::Bool(true),
            GroupKey::Num {
                raw: b"2".to_vec(),
                bits: order_bits(2.0),
            },
            GroupKey::Bool(false),
        ];
        keys.sort();
        assert_eq!(
            keys,
            vec![
                GroupKey::Null,
                GroupKey::Bool(false),
                GroupKey::Bool(true),
                GroupKey::Num {
                    raw: b"2".to_vec(),
                    bits: order_bits(2.0)
                },
                GroupKey::Num {
                    raw: b"10".to_vec(),
                    bits: order_bits(10.0)
                },
                GroupKey::Str(b"a".to_vec()),
                GroupKey::Str(b"b".to_vec()),
            ]
        );
    }

    #[test]
    fn numeric_group_keys_sort_numerically_not_lexically() {
        // The classic bug: "10" sorting before "2".
        let mut keys = [
            GroupKey::Num {
                raw: b"10".to_vec(),
                bits: order_bits(10.0),
            },
            GroupKey::Num {
                raw: b"2".to_vec(),
                bits: order_bits(2.0),
            },
            GroupKey::Num {
                raw: b"-5".to_vec(),
                bits: order_bits(-5.0),
            },
        ];
        keys.sort();
        let raws: Vec<_> = keys
            .iter()
            .map(|k| match k {
                GroupKey::Num { raw, .. } => String::from_utf8(raw.clone()).unwrap(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(raws, ["-5", "2", "10"]);
    }

    #[test]
    fn number_formatting_matches_jq_conventions() {
        assert_eq!(format_number(1.0), "1");
        assert_eq!(format_number(-0.0), "0");
        assert_eq!(format_number(1.5), "1.5");
        assert_eq!(format_number(1e300), "1e+300");
        assert_eq!(format_number(1e-300), "1e-300");
        assert_eq!(format_number(0.1 + 0.2), "0.30000000000000004");
        // Past 2^53 integers stop being exact, so we stop pretending.
        assert!(format_number(1e18).contains('e'));
    }

    #[test]
    fn group_accumulator_merge_is_order_independent() {
        let mut a = GroupAccumulator::default();
        a.entry(GroupKey::Str(b"x".to_vec())).push_number(1.0);
        let mut b = GroupAccumulator::default();
        b.entry(GroupKey::Str(b"x".to_vec())).push_number(2.0);
        b.entry(GroupKey::Str(b"y".to_vec())).push_number(5.0);
        a.merge(b);
        let sorted = a.sorted();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].1.sum, 3.0);
        assert_eq!(sorted[1].1.sum, 5.0);
    }
}
