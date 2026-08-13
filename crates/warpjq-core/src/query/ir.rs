//! The compiled representation of a query.
//!
//! Both backends consume `Program` and nothing else. That is the whole point:
//! there is exactly one place where "what does this query mean" is decided, so
//! the GPU cannot silently disagree with the CPU. The differential tests in
//! `tests/differential.rs` lean on this.

use std::fmt;

/// Index into [`Program::paths`].
pub type PathId = u16;

/// One hop in a field path. `.a.b[2]` is `[Key("a"), Key("b"), Index(2)]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Key(String),
    Index(u32),
}

/// A field path. An empty step list is jq's identity, `.`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Path {
    pub steps: Vec<Step>,
}

impl Path {
    pub fn is_identity(&self) -> bool {
        self.steps.is_empty()
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.steps.is_empty() {
            return write!(f, ".");
        }
        for s in &self.steps {
            match s {
                Step::Key(k) if is_bare_key(k) => write!(f, ".{k}")?,
                Step::Key(k) => write!(f, ".{k:?}")?,
                Step::Index(i) => write!(f, "[{i}]")?,
            }
        }
        Ok(())
    }
}

pub(crate) fn is_bare_key(k: &str) -> bool {
    !k.is_empty()
        && !k.as_bytes()[0].is_ascii_digit()
        && k.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// Apply to the result of a jq-style three-way comparison.
    pub fn accepts(self, ordering: std::cmp::Ordering) -> bool {
        use std::cmp::Ordering::*;
        match self {
            CmpOp::Eq => ordering == Equal,
            CmpOp::Ne => ordering != Equal,
            CmpOp::Lt => ordering == Less,
            CmpOp::Le => ordering != Greater,
            CmpOp::Gt => ordering == Greater,
            CmpOp::Ge => ordering != Less,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
}

impl Literal {
    /// jq's total order across types: null < false < true < numbers < strings.
    pub fn type_rank(&self) -> u8 {
        match self {
            Literal::Null => 0,
            Literal::Bool(false) => 1,
            Literal::Bool(true) => 2,
            Literal::Num(_) => 3,
            Literal::Str(_) => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Cond {
    /// `.path <op> literal`
    Cmp {
        path: PathId,
        op: CmpOp,
        lit: Literal,
    },
    /// `select(.path)`: true unless the value is missing, `null`, or `false`.
    Truthy(PathId),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Not(Box<Cond>),
}

impl Cond {
    /// Every path this condition touches, for path-set collection.
    pub fn visit_paths(&self, f: &mut impl FnMut(PathId)) {
        match self {
            Cond::Cmp { path, .. } | Cond::Truthy(path) => f(*path),
            Cond::And(a, b) | Cond::Or(a, b) => {
                a.visit_paths(f);
                b.visit_paths(f);
            }
            Cond::Not(a) => a.visit_paths(f),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AggKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

impl AggKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AggKind::Count => "count",
            AggKind::Sum => "sum",
            AggKind::Min => "min",
            AggKind::Max => "max",
            AggKind::Avg => "avg",
        }
    }

    /// `count` is the only aggregate that takes no argument.
    pub fn needs_arg(self) -> bool {
        self != AggKind::Count
    }
}

/// What a surviving line turns into.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// Emit the input line verbatim. Byte-preserving, with no reformatting, so
    /// big integers and unusual-but-valid number spellings round-trip exactly.
    Passthrough,
    /// Emit one extracted value, as raw JSON.
    Path(PathId),
    /// `{a: .x, b: .y}`. Key order is the order written in the query.
    Project(Vec<(String, PathId)>),
    /// A whole-stream reduction. Emits exactly one line (or one per group).
    Agg { kind: AggKind, arg: Option<PathId> },
}

/// A fully compiled query.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// Interned paths. Both backends index into this by [`PathId`].
    pub paths: Vec<Path>,
    /// `select(...)`, if any. `None` means every line survives.
    pub filter: Option<Cond>,
    /// `group_by(.path)`, if any.
    pub group_by: Option<PathId>,
    pub output: Output,
    /// The query text, kept for error messages and `--bench` labels.
    pub source: String,
}

impl Program {
    pub fn path(&self, id: PathId) -> &Path {
        &self.paths[id as usize]
    }

    /// True when the result is a single row (or one row per group) rather
    /// than a stream. Drives buffering decisions in the pipeline.
    pub fn is_aggregate(&self) -> bool {
        matches!(self.output, Output::Agg { .. })
    }

    /// The paths that must be extracted per line, deduplicated.
    pub fn required_paths(&self) -> Vec<PathId> {
        let mut ids = Vec::new();
        let mut push = |id: PathId| {
            if !ids.contains(&id) {
                ids.push(id);
            }
        };
        if let Some(c) = &self.filter {
            c.visit_paths(&mut push);
        }
        if let Some(g) = self.group_by {
            push(g);
        }
        match &self.output {
            Output::Passthrough => {}
            Output::Path(p) => push(*p),
            Output::Project(fields) => {
                for (_, p) in fields {
                    push(*p);
                }
            }
            Output::Agg { arg, .. } => {
                if let Some(p) = arg {
                    push(*p);
                }
            }
        }
        ids
    }

    /// Column headers for `--csv`, when the shape is known statically.
    pub fn csv_headers(&self) -> Option<Vec<String>> {
        match &self.output {
            Output::Project(fields) => Some(fields.iter().map(|(k, _)| k.clone()).collect()),
            Output::Path(p) => Some(vec![self.path(*p).to_string()]),
            Output::Agg { kind, .. } => {
                let mut cols = Vec::new();
                if let Some(g) = self.group_by {
                    cols.push(self.path(g).to_string());
                }
                cols.push(kind.as_str().to_string());
                Some(cols)
            }
            Output::Passthrough => None,
        }
    }
}
