//! Query front-end: text -> [`Program`] -> flat tables the GPU can consume.

pub mod ir;
mod lex;
mod parse;

pub use ir::*;
pub use parse::parse;

/// FNV-1a 32-bit. Used as a cheap prefilter before the byte-wise key compare.
///
/// The device kernel computes the same value; if you change this, change
/// `warpjq_key_hash` in `cuda/warpjq_kernels.cu` in the same commit or key
/// lookup silently starts missing.
pub fn key_hash(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

pub const STEP_KEY: u32 = 0;
pub const STEP_INDEX: u32 = 1;

pub const LIT_NULL: u32 = 0;
pub const LIT_FALSE: u32 = 1;
pub const LIT_TRUE: u32 = 2;
pub const LIT_NUM: u32 = 3;
pub const LIT_STR: u32 = 4;

pub const COND_CMP: u32 = 0;
pub const COND_TRUTHY: u32 = 1;
pub const COND_AND: u32 = 2;
pub const COND_OR: u32 = 3;
pub const COND_NOT: u32 = 4;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuStep {
    pub kind: u32,
    pub index: u32,
    pub key_off: u32,
    pub key_len: u32,
    pub key_hash: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuPath {
    pub step_off: u32,
    pub step_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuCmp {
    pub path: u32,
    pub op: u32,
    pub lit_kind: u32,
    pub lit_off: u32,
    pub lit_len: u32,
    pub _pad: u32,
    pub lit_num: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuCondOp {
    pub op: u32,
    /// Index into `cmps` for `COND_CMP`, into `paths` for `COND_TRUTHY`,
    /// unused otherwise.
    pub arg: u32,
}

/// The flat, pointer-free form of a [`Program`] that gets uploaded once per run.
///
/// Everything is an index into a side table so the whole thing is a handful of
/// small `cudaMemcpy`s and the kernel needs no dynamic allocation.
#[derive(Clone, Debug, Default)]
pub struct FlatProgram {
    pub steps: Vec<GpuStep>,
    pub paths: Vec<GpuPath>,
    pub cmps: Vec<GpuCmp>,
    /// Condition in reverse-Polish order; evaluate with a tiny boolean stack.
    pub cond_rpn: Vec<GpuCondOp>,
    /// Concatenated key names and string literals, referenced by offset/len.
    pub blob: Vec<u8>,
    /// Maximum boolean-stack depth needed to evaluate `cond_rpn`.
    pub cond_stack_depth: u32,
}

impl FlatProgram {
    pub fn build(p: &Program) -> Self {
        let mut f = FlatProgram::default();

        for path in &p.paths {
            let step_off = f.steps.len() as u32;
            for step in &path.steps {
                let gs = match step {
                    Step::Key(k) => {
                        let (off, len) = f.push_blob(k.as_bytes());
                        GpuStep {
                            kind: STEP_KEY,
                            index: 0,
                            key_off: off,
                            key_len: len,
                            key_hash: key_hash(k.as_bytes()),
                        }
                    }
                    Step::Index(i) => GpuStep {
                        kind: STEP_INDEX,
                        index: *i,
                        ..Default::default()
                    },
                };
                f.steps.push(gs);
            }
            f.paths.push(GpuPath {
                step_off,
                step_count: path.steps.len() as u32,
            });
        }

        if let Some(cond) = &p.filter {
            f.cond_stack_depth = flatten_cond(cond, &mut f);
        }

        f
    }

    fn push_blob(&mut self, bytes: &[u8]) -> (u32, u32) {
        let off = self.blob.len() as u32;
        self.blob.extend_from_slice(bytes);
        (off, bytes.len() as u32)
    }
}

/// Emits `cond` in RPN and returns the peak boolean-stack depth it needs.
fn flatten_cond(cond: &Cond, f: &mut FlatProgram) -> u32 {
    match cond {
        Cond::Cmp { path, op, lit } => {
            let (lit_kind, lit_num, lit_off, lit_len) = match lit {
                Literal::Null => (LIT_NULL, 0.0, 0, 0),
                Literal::Bool(false) => (LIT_FALSE, 0.0, 0, 0),
                Literal::Bool(true) => (LIT_TRUE, 0.0, 0, 0),
                Literal::Num(n) => (LIT_NUM, *n, 0, 0),
                Literal::Str(s) => {
                    let (off, len) = f.push_blob(s.as_bytes());
                    (LIT_STR, 0.0, off, len)
                }
            };
            let idx = f.cmps.len() as u32;
            f.cmps.push(GpuCmp {
                path: *path as u32,
                op: *op as u32,
                lit_kind,
                lit_off,
                lit_len,
                _pad: 0,
                lit_num,
            });
            f.cond_rpn.push(GpuCondOp {
                op: COND_CMP,
                arg: idx,
            });
            1
        }
        Cond::Truthy(p) => {
            f.cond_rpn.push(GpuCondOp {
                op: COND_TRUTHY,
                arg: *p as u32,
            });
            1
        }
        Cond::Not(a) => {
            let d = flatten_cond(a, f);
            f.cond_rpn.push(GpuCondOp {
                op: COND_NOT,
                arg: 0,
            });
            d
        }
        Cond::And(a, b) | Cond::Or(a, b) => {
            let da = flatten_cond(a, f);
            let db = flatten_cond(b, f);
            let op = if matches!(cond, Cond::And(_, _)) {
                COND_AND
            } else {
                COND_OR
            };
            f.cond_rpn.push(GpuCondOp { op, arg: 0 });
            // Left subtree's result occupies one slot while the right evaluates.
            da.max(db + 1)
        }
    }
}

/// `CmpOp` is `#[repr]`-less, so pin the discriminants the GPU relies on.
/// A unit test asserts these match.
pub const OP_EQ: u32 = 0;
pub const OP_NE: u32 = 1;
pub const OP_LT: u32 = 2;
pub const OP_LE: u32 = 3;
pub const OP_GT: u32 = 4;
pub const OP_GE: u32 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmp_op_discriminants_match_cuda_constants() {
        assert_eq!(CmpOp::Eq as u32, OP_EQ);
        assert_eq!(CmpOp::Ne as u32, OP_NE);
        assert_eq!(CmpOp::Lt as u32, OP_LT);
        assert_eq!(CmpOp::Le as u32, OP_LE);
        assert_eq!(CmpOp::Gt as u32, OP_GT);
        assert_eq!(CmpOp::Ge as u32, OP_GE);
    }

    fn prog(src: &str) -> Program {
        parse(src).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    #[test]
    fn parses_identity() {
        let p = prog(".");
        assert_eq!(p.output, Output::Passthrough);
        assert!(p.filter.is_none());
    }

    #[test]
    fn parses_nested_and_indexed_paths() {
        let p = prog(".a.b[2].c");
        assert_eq!(
            p.paths[0].steps,
            vec![
                Step::Key("a".into()),
                Step::Key("b".into()),
                Step::Index(2),
                Step::Key("c".into())
            ]
        );
    }

    #[test]
    fn parses_quoted_keys() {
        let p = prog(r#"."content-type""#);
        assert_eq!(p.paths[0].steps, vec![Step::Key("content-type".into())]);
    }

    #[test]
    fn parses_select_and_count() {
        let p = prog("select(.status == 500) | count");
        assert!(matches!(p.filter, Some(Cond::Cmp { .. })));
        assert_eq!(
            p.output,
            Output::Agg {
                kind: AggKind::Count,
                arg: None
            }
        );
    }

    #[test]
    fn conjoins_multiple_selects() {
        let p = prog("select(.a == 1) | select(.b == 2)");
        assert!(matches!(p.filter, Some(Cond::And(_, _))));
    }

    #[test]
    fn and_binds_tighter_than_or() {
        let p = prog("select(.a == 1 or .b == 2 and .c == 3)");
        match p.filter.unwrap() {
            Cond::Or(_, rhs) => assert!(matches!(*rhs, Cond::And(_, _))),
            other => panic!("expected Or at the root, got {other:?}"),
        }
    }

    #[test]
    fn parses_projection_with_shorthand() {
        let p = prog("{time: .ts, msg}");
        match &p.output {
            Output::Project(f) => {
                assert_eq!(f[0].0, "time");
                assert_eq!(f[1].0, "msg");
                assert_eq!(p.path(f[1].1).steps, vec![Step::Key("msg".into())]);
            }
            o => panic!("expected a projection, got {o:?}"),
        }
    }

    #[test]
    fn parses_group_by() {
        let p = prog("group_by(.host) | count");
        assert!(p.group_by.is_some());
        assert!(p.is_aggregate());
    }

    #[test]
    fn parses_postfix_not() {
        let p = prog("select(.ok | not)");
        assert!(matches!(p.filter, Some(Cond::Not(_))));
    }

    #[test]
    fn interns_repeated_paths() {
        let p = prog("select(.a == 1 or .a == 2) | {x: .a}");
        assert_eq!(p.paths.len(), 1);
    }

    #[test]
    fn required_paths_are_deduplicated() {
        let p = prog("select(.a == 1) | {x: .a, y: .b}");
        assert_eq!(p.required_paths().len(), 2);
    }

    #[test]
    fn rejects_unsupported_constructs_by_name() {
        for q in ["reduce", "map", "if", "def", "test", "length"] {
            let e = parse(q).unwrap_err();
            assert!(
                e.message.contains("not in the v0.1 subset") || e.message.contains("roadmap"),
                "{q} gave: {}",
                e.message
            );
        }
    }

    #[test]
    fn names_unsupported_constructs_even_when_their_syntax_does_not_lex() {
        // `reduce ... as $x` contains a `$`, which the lexer rejects outright.
        // Reporting "unexpected character `$`" would tell the user nothing
        // about why their query is unsupported, so the keyword is recognised
        // before lexing.
        for (q, want) in [
            ("reduce .[] as $x (0; .+$x)", "`reduce` is not"),
            ("def f: .; f", "user-defined functions"),
            ("if .a then 1 else 2 end", "`if/then/else` is not"),
            ("foreach .[] as $x (0; .+1)", "`foreach` is not"),
        ] {
            let e = parse(q).unwrap_err();
            assert!(e.message.contains(want), "{q} gave: {}", e.message);
        }
    }

    #[test]
    fn names_unsupported_constructs_in_later_pipeline_stages() {
        for q in [
            "select(.a == 1) | map(.b)",
            r#".a | test("x")"#,
            "select(.a == 1) | keys",
        ] {
            let e = parse(q).unwrap_err();
            assert!(
                e.message.contains("not in the v0.1 subset") || e.message.contains("roadmap"),
                "{q} gave: {}",
                e.message
            );
        }
    }

    #[test]
    fn a_pipe_inside_a_string_or_parens_does_not_split_a_stage() {
        // The stage splitter must not treat these as pipeline separators, or
        // it would look for keywords in the middle of an expression.
        assert!(parse(r#"select(.msg == "a|b")"#).is_ok());
        assert!(parse("select(.a == 1 or (.b == 2))").is_ok());
        // And `not` after a real top-level pipe still works.
        assert!(parse("select(.a | not)").is_ok());
    }

    #[test]
    fn rejects_assignment_with_a_hint() {
        let e = parse("select(.a = 1)").unwrap_err();
        assert!(e.message.contains("did you mean `==`"));
    }

    #[test]
    fn rejects_missing_dot_with_a_hint() {
        let e = parse("select(status == 500)").unwrap_err();
        assert_eq!(e.hint.as_deref(), Some("did you mean `.status`?"));
    }

    #[test]
    fn rejects_group_by_without_aggregate() {
        assert!(parse("group_by(.host)").is_err());
        assert!(parse("group_by(.host) | {a: .b}").is_err());
    }

    #[test]
    fn rejects_field_to_field_comparison_clearly() {
        let e = parse("select(.a == .b)").unwrap_err();
        assert!(e.message.contains("field-to-field"));
    }

    #[test]
    fn rejects_two_output_stages() {
        assert!(parse("count | count").is_err());
        assert!(parse("{a: .b} | {c: .d}").is_err());
    }

    #[test]
    fn flattens_and_or_to_rpn_with_correct_depth() {
        let p = prog("select((.a == 1 and .b == 2) or .c == 3)");
        let f = FlatProgram::build(&p);
        assert_eq!(f.cmps.len(), 3);
        // cmp cmp AND cmp OR
        assert_eq!(f.cond_rpn.len(), 5);
        assert_eq!(f.cond_rpn[2].op, COND_AND);
        assert_eq!(f.cond_rpn[4].op, COND_OR);
        assert_eq!(f.cond_stack_depth, 2);
    }

    #[test]
    fn flat_string_literals_land_in_the_blob() {
        let p = prog(r#"select(.method == "POST")"#);
        let f = FlatProgram::build(&p);
        let c = f.cmps[0];
        assert_eq!(c.lit_kind, LIT_STR);
        let bytes = &f.blob[c.lit_off as usize..(c.lit_off + c.lit_len) as usize];
        assert_eq!(bytes, b"POST");
    }

    #[test]
    fn path_display_round_trips() {
        assert_eq!(prog(".a.b[0]").paths[0].to_string(), ".a.b[0]");
        assert_eq!(prog(".").output, Output::Passthrough);
    }
}
