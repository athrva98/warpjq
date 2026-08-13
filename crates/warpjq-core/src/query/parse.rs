//! Recursive-descent parser for the supported jq subset.
//!
//! Grammar (v0.1):
//!
//! ```text
//! program  := stage ('|' stage)*
//! stage    := 'select' '(' cond ')'
//!           | 'group_by' '(' path ')'
//!           | 'count'
//!           | ('sum'|'min'|'max'|'avg') '(' path ')'
//!           | object
//!           | path
//! object   := '{' [ entry (',' entry)* [','] ] '}'
//! entry    := (ident | string) ':' path | ident          # {foo} == {foo: .foo}
//! cond     := or_expr
//! or_expr  := and_expr ('or' and_expr)*
//! and_expr := term ('and' term)*
//! term     := '(' cond ')' ['|' 'not']
//!           | path [op literal] ['|' 'not']
//! op       := '==' | '!=' | '<' | '<=' | '>' | '>='
//! path     := '.' | ('.' ident | '.' string | '.' '[' int ']' | '[' int ']')+
//! literal  := number | string | 'true' | 'false' | 'null'
//! ```
//!
//! Anything outside this is rejected with a pointed error rather than being
//! silently misinterpreted. Refusing `reduce`/`def`/`//`/regex loudly is the
//! honest thing to do when the README claims a subset.

use super::ir::*;
use super::lex::{lex, Spanned, Tok};
use crate::error::{QueryError, QueryResult};

/// Constructs recognised by real jq that we deliberately do not implement.
/// Naming them individually turns "syntax error" into "not supported yet".
const KNOWN_UNSUPPORTED: &[(&str, &str)] = &[
    ("reduce", "`reduce` is not in the v0.1 subset"),
    ("foreach", "`foreach` is not in the v0.1 subset"),
    ("def", "user-defined functions are not in the v0.1 subset"),
    (
        "if",
        "`if/then/else` is not in the v0.1 subset; use `select(...)`",
    ),
    ("try", "`try/catch` is not in the v0.1 subset"),
    ("test", "regex (`test`) is on the roadmap, not in v0.1"),
    ("match", "regex (`match`) is on the roadmap, not in v0.1"),
    ("map", "`map` is not in the v0.1 subset"),
    ("keys", "`keys` is not in the v0.1 subset"),
    (
        "length",
        "`length` is not in the v0.1 subset; `count` counts lines",
    ),
    ("tostring", "type coercions are not in the v0.1 subset"),
    ("tonumber", "type coercions are not in the v0.1 subset"),
    ("sort_by", "`sort_by` is not in the v0.1 subset"),
    ("unique", "`unique` is not in the v0.1 subset"),
    ("add", "`add` is not in the v0.1 subset; use `sum(.field)`"),
];

pub fn parse(src: &str) -> QueryResult<Program> {
    // Check the leading keyword before lexing. `reduce .[] as $x (0; .+$x)`
    // would otherwise die on the `$` with "unexpected character", which tells
    // the user nothing about why their query is not supported.
    if let Some(e) = unsupported_leading_keyword(src) {
        return Err(e);
    }
    let toks = lex(src)?;
    let mut p = Parser {
        src,
        toks,
        pos: 0,
        paths: Vec::new(),
    };
    p.program()
}

/// Recognises an unsupported jq construct at the start of the query, or at the
/// start of any pipeline stage, before the lexer can trip over syntax that
/// only exists inside that construct.
fn unsupported_leading_keyword(src: &str) -> Option<QueryError> {
    for (stage_start, stage) in stage_starts(src) {
        let trimmed = stage.trim_start();
        let lead_off = stage_start + (stage.len() - trimmed.len());
        let word: String = trimmed
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if word.is_empty() {
            continue;
        }
        if let Some((_, why)) = KNOWN_UNSUPPORTED.iter().find(|(n, _)| *n == word) {
            return Some(
                QueryError::new(src, lead_off, *why).with_hint(
                    "see the Limitations section of the README for the full v0.1 subset",
                ),
            );
        }
    }
    None
}

/// Splits on top-level `|`, ignoring pipes inside strings, brackets or parens.
fn stage_starts(src: &str) -> Vec<(usize, &str)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut i = 0usize;
    while i < b.len() {
        match b[i] {
            b'\\' if in_str => i += 1,
            b'"' => in_str = !in_str,
            b'(' | b'[' | b'{' if !in_str => depth += 1,
            b')' | b']' | b'}' if !in_str => depth -= 1,
            b'|' if !in_str && depth == 0 => {
                out.push((start, &src[start..i]));
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push((start, &src[start..]));
    out
}

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Spanned>,
    pos: usize,
    paths: Vec<Path>,
}

impl<'a> Parser<'a> {
    // ---- token helpers -------------------------------------------------

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn at(&self) -> usize {
        self.toks[self.pos].at
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, ctx: &str) -> QueryResult<()> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(self.err(format!(
                "expected {} {ctx}, found {}",
                t.describe(),
                self.peek().describe()
            )))
        }
    }

    fn err(&self, msg: impl Into<String>) -> QueryError {
        QueryError::new(self.src, self.at(), msg)
    }

    fn peek_keyword(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    /// Interning keeps `PathId` small and lets the GPU upload one path table.
    fn intern(&mut self, path: Path) -> PathId {
        if let Some(i) = self.paths.iter().position(|p| *p == path) {
            return i as PathId;
        }
        self.paths.push(path);
        (self.paths.len() - 1) as PathId
    }

    // ---- grammar -------------------------------------------------------

    fn program(&mut self) -> QueryResult<Program> {
        let mut filter: Option<Cond> = None;
        let mut group_by: Option<PathId> = None;
        let mut output: Option<Output> = None;
        let mut output_at = 0usize;

        loop {
            let stage_at = self.at();
            match self.stage()? {
                Stage::Filter(c) => {
                    // Multiple `select`s in a pipeline conjoin, exactly as in jq.
                    filter = Some(match filter {
                        None => c,
                        Some(prev) => Cond::And(Box::new(prev), Box::new(c)),
                    });
                }
                Stage::GroupBy(p) => {
                    if group_by.is_some() {
                        return Err(QueryError::new(
                            self.src,
                            stage_at,
                            "only one `group_by` per query is supported in v0.1",
                        ));
                    }
                    if output.is_some() {
                        return Err(QueryError::new(
                            self.src,
                            stage_at,
                            "`group_by` must come before the output stage",
                        )
                        .with_hint(
                            "write `group_by(.host) | count`, not `count | group_by(.host)`",
                        ));
                    }
                    group_by = Some(p);
                }
                Stage::Output(o) => {
                    if let Some(prev) = &output {
                        return Err(QueryError::new(
                            self.src,
                            stage_at,
                            format!(
                                "a second output stage follows {}; v0.1 supports one \
                                 output stage per query",
                                describe_output(prev)
                            ),
                        ));
                    }
                    output = Some(o);
                    output_at = stage_at;
                }
            }

            if !self.eat(&Tok::Pipe) {
                break;
            }
        }

        if !matches!(self.peek(), Tok::Eof) {
            return Err(self.err(format!("unexpected {}", self.peek().describe())));
        }

        let output = output.unwrap_or(Output::Passthrough);

        // `group_by` only means anything with an aggregate after it.
        if group_by.is_some() && !matches!(output, Output::Agg { .. }) {
            return Err(QueryError::new(
                self.src,
                output_at.max(1),
                "`group_by` must be followed by an aggregate",
            )
            .with_hint("try `group_by(.host) | count` or `group_by(.host) | sum(.bytes)`"));
        }

        Ok(Program {
            paths: std::mem::take(&mut self.paths),
            filter,
            group_by,
            output,
            source: self.src.to_string(),
        })
    }

    fn stage(&mut self) -> QueryResult<Stage> {
        let at = self.at();

        if let Tok::Ident(name) = self.peek().clone() {
            if let Some((_, why)) = KNOWN_UNSUPPORTED.iter().find(|(n, _)| *n == name) {
                return Err(QueryError::new(self.src, at, *why).with_hint(
                    "see the Limitations section of the README for the full v0.1 subset",
                ));
            }
            match name.as_str() {
                "select" => {
                    self.bump();
                    self.expect(&Tok::LParen, "after `select`")?;
                    let c = self.cond()?;
                    self.expect(&Tok::RParen, "to close `select(`")?;
                    return Ok(Stage::Filter(c));
                }
                "group_by" => {
                    self.bump();
                    self.expect(&Tok::LParen, "after `group_by`")?;
                    let p = self.path()?;
                    if p.is_identity() {
                        return Err(QueryError::new(
                            self.src,
                            at,
                            "`group_by(.)` would group by the whole line",
                        )
                        .with_hint("group by a field, e.g. `group_by(.status)`"));
                    }
                    let id = self.intern(p);
                    self.expect(&Tok::RParen, "to close `group_by(`")?;
                    return Ok(Stage::GroupBy(id));
                }
                "count" => {
                    self.bump();
                    // `count(...)` is not jq; catch it before it becomes a
                    // confusing "unexpected `(`".
                    if matches!(self.peek(), Tok::LParen) {
                        return Err(self
                            .err("`count` takes no argument")
                            .with_hint("write `count`, or `sum(.field)` to total a field"));
                    }
                    return Ok(Stage::Output(Output::Agg {
                        kind: AggKind::Count,
                        arg: None,
                    }));
                }
                "sum" | "min" | "max" | "avg" => {
                    self.bump();
                    let kind = match name.as_str() {
                        "sum" => AggKind::Sum,
                        "min" => AggKind::Min,
                        "max" => AggKind::Max,
                        _ => AggKind::Avg,
                    };
                    self.expect(&Tok::LParen, &format!("after `{name}`"))?;
                    let p = self.path()?;
                    if p.is_identity() {
                        return Err(QueryError::new(
                            self.src,
                            at,
                            format!("`{name}(.)` has no field to aggregate"),
                        )
                        .with_hint(format!("try `{name}(.bytes)`")));
                    }
                    let id = self.intern(p);
                    self.expect(&Tok::RParen, &format!("to close `{name}(`"))?;
                    return Ok(Stage::Output(Output::Agg {
                        kind,
                        arg: Some(id),
                    }));
                }
                _ => {
                    return Err(
                        QueryError::new(self.src, at, format!("unknown function `{name}`"))
                            .with_hint(
                                "v0.1 supports: select, group_by, count, sum, min, max, avg",
                            ),
                    )
                }
            }
        }

        if matches!(self.peek(), Tok::LBrace) {
            return Ok(Stage::Output(self.object()?));
        }

        let p = self.path()?;
        if p.is_identity() {
            return Ok(Stage::Output(Output::Passthrough));
        }
        let id = self.intern(p);
        Ok(Stage::Output(Output::Path(id)))
    }

    fn object(&mut self) -> QueryResult<Output> {
        self.expect(&Tok::LBrace, "to start an object")?;
        let mut fields: Vec<(String, PathId)> = Vec::new();
        loop {
            if matches!(self.peek(), Tok::RBrace) {
                break;
            }
            let key_at = self.at();
            let key = match self.bump() {
                Tok::Ident(s) => s,
                Tok::Str(s) => s,
                other => {
                    return Err(QueryError::new(
                        self.src,
                        key_at,
                        format!("expected an object key, found {}", other.describe()),
                    ))
                }
            };
            let path_id = if self.eat(&Tok::Colon) {
                let p = self.path()?;
                if p.is_identity() {
                    return Err(QueryError::new(
                        self.src,
                        key_at,
                        "an object value of `.` would embed the whole line",
                    )
                    .with_hint("name a field, e.g. `{msg: .message}`"));
                }
                self.intern(p)
            } else {
                // jq's `{foo}` shorthand for `{foo: .foo}`.
                self.intern(Path {
                    steps: vec![Step::Key(key.clone())],
                })
            };
            if fields.iter().any(|(k, _)| *k == key) {
                return Err(QueryError::new(
                    self.src,
                    key_at,
                    format!("duplicate key `{key}` in projection"),
                ));
            }
            fields.push((key, path_id));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RBrace, "to close the object")?;
        if fields.is_empty() {
            return Err(self.err("empty projection `{}` produces no output"));
        }
        Ok(Output::Project(fields))
    }

    // ---- conditions ----------------------------------------------------

    fn cond(&mut self) -> QueryResult<Cond> {
        let mut lhs = self.and_expr()?;
        while self.peek_keyword("or") {
            self.bump();
            let rhs = self.and_expr()?;
            lhs = Cond::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn and_expr(&mut self) -> QueryResult<Cond> {
        let mut lhs = self.term()?;
        while self.peek_keyword("and") {
            self.bump();
            let rhs = self.term()?;
            lhs = Cond::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn term(&mut self) -> QueryResult<Cond> {
        let at = self.at();

        let base = if self.eat(&Tok::LParen) {
            let c = self.cond()?;
            self.expect(&Tok::RParen, "to close the group")?;
            c
        } else {
            if let Tok::Ident(name) = self.peek().clone() {
                // A bare identifier here is almost always a forgotten dot.
                return Err(QueryError::new(
                    self.src,
                    at,
                    format!("expected a field path, found `{name}`"),
                )
                .with_hint(format!("did you mean `.{name}`?")));
            }
            let p = self.path()?;
            if p.is_identity() {
                return Err(QueryError::new(
                    self.src,
                    at,
                    "`.` cannot be used as a condition in v0.1",
                )
                .with_hint("compare a field, e.g. `select(.status == 500)`"));
            }
            let id = self.intern(p);

            if let Tok::Op(op) = self.peek().clone() {
                self.bump();
                let lit = self.literal()?;
                Cond::Cmp { path: id, op, lit }
            } else {
                // `select(.error)`: truthiness, as in jq.
                Cond::Truthy(id)
            }
        };

        // Postfix `| not`, which is how jq spells negation.
        let mut out = base;
        while matches!(self.peek(), Tok::Pipe)
            && matches!(self.peek_at(1), Tok::Ident(s) if s == "not")
        {
            self.bump();
            self.bump();
            out = Cond::Not(Box::new(out));
        }
        Ok(out)
    }

    fn literal(&mut self) -> QueryResult<Literal> {
        let at = self.at();
        Ok(match self.bump() {
            Tok::Num(n) => Literal::Num(n),
            Tok::Int(n) => Literal::Num(n as f64),
            Tok::Str(s) => Literal::Str(s),
            Tok::Ident(s) => match s.as_str() {
                "true" => Literal::Bool(true),
                "false" => Literal::Bool(false),
                "null" => Literal::Null,
                _ => {
                    return Err(QueryError::new(
                        self.src,
                        at,
                        format!("expected a literal value, found `{s}`"),
                    )
                    .with_hint(
                        "the right-hand side of a comparison must be a number, string, \
                         `true`, `false`, or `null`. Field-to-field comparison is not \
                         in v0.1",
                    ))
                }
            },
            Tok::Dot => {
                return Err(QueryError::new(
                    self.src,
                    at,
                    "field-to-field comparison is not supported in v0.1",
                )
                .with_hint("compare against a constant, e.g. `select(.status == 500)`"))
            }
            other => {
                return Err(QueryError::new(
                    self.src,
                    at,
                    format!("expected a literal value, found {}", other.describe()),
                ))
            }
        })
    }

    // ---- paths ---------------------------------------------------------

    fn path(&mut self) -> QueryResult<Path> {
        let at = self.at();
        if !matches!(self.peek(), Tok::Dot) {
            return Err(QueryError::new(
                self.src,
                at,
                format!("expected a field path, found {}", self.peek().describe()),
            )
            .with_hint("paths start with `.`, e.g. `.status` or `.req.path`"));
        }
        self.bump();

        let mut steps = Vec::new();
        // A lone `.` is identity; `.foo`, `.[0]`, `."odd key"` continue.
        loop {
            match self.peek().clone() {
                Tok::Ident(name) => {
                    self.bump();
                    steps.push(Step::Key(name));
                }
                Tok::Str(s) => {
                    self.bump();
                    steps.push(Step::Key(s));
                }
                Tok::LBracket => {
                    self.bump();
                    let idx_at = self.at();
                    match self.bump() {
                        Tok::Int(i) if i >= 0 => steps.push(Step::Index(i as u32)),
                        Tok::Int(_) => {
                            return Err(QueryError::new(
                                self.src,
                                idx_at,
                                "negative array indices are not supported in v0.1",
                            ))
                        }
                        other => {
                            return Err(QueryError::new(
                                self.src,
                                idx_at,
                                format!("expected an array index, found {}", other.describe()),
                            )
                            .with_hint("slices (`.[1:3]`) and iteration (`.[]`) are not in v0.1"))
                        }
                    }
                    self.expect(&Tok::RBracket, "to close the index")?;
                }
                _ => break,
            }
            // Between steps a `.` is optional: `.a.b[0].c` and `.a.b[0]c` are
            // not both legal jq, so require the dot except before `[`.
            if matches!(self.peek(), Tok::Dot) {
                // Lookahead: `.` followed by something path-like continues.
                match self.peek_at(1) {
                    Tok::Ident(_) | Tok::Str(_) | Tok::LBracket => {
                        self.bump();
                    }
                    _ => break,
                }
            } else if !matches!(self.peek(), Tok::LBracket) {
                break;
            }
        }

        Ok(Path { steps })
    }
}

enum Stage {
    Filter(Cond),
    GroupBy(PathId),
    Output(Output),
}

fn describe_output(o: &Output) -> &'static str {
    match o {
        Output::Passthrough => "`.`",
        Output::Path(_) => "a field path",
        Output::Project(_) => "an object projection",
        Output::Agg { .. } => "an aggregate",
    }
}
