//! Tokeniser for the supported jq subset.

use crate::error::{QueryError, QueryResult};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Dot,
    Pipe,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Op(crate::query::ir::CmpOp),
    Ident(String),
    Str(String),
    Num(f64),
    /// Integer spelling preserved, for `[3]` indices.
    Int(i64),
    Eof,
}

impl Tok {
    pub fn describe(&self) -> String {
        match self {
            Tok::Dot => "`.`".into(),
            Tok::Pipe => "`|`".into(),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::LBracket => "`[`".into(),
            Tok::RBracket => "`]`".into(),
            Tok::LBrace => "`{`".into(),
            Tok::RBrace => "`}`".into(),
            Tok::Colon => "`:`".into(),
            Tok::Comma => "`,`".into(),
            Tok::Op(o) => format!("`{}`", o.as_str()),
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Str(s) => format!("string {s:?}"),
            Tok::Num(n) => format!("number `{n}`"),
            Tok::Int(n) => format!("number `{n}`"),
            Tok::Eof => "end of query".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Spanned {
    pub tok: Tok,
    /// Byte offset of the token start, for caret diagnostics.
    pub at: usize,
}

pub fn lex(src: &str) -> QueryResult<Vec<Spanned>> {
    use crate::query::ir::CmpOp;
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // `#` comments, as in jq.
        if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let at = i;
        let tok = match c {
            b'.' => {
                i += 1;
                Tok::Dot
            }
            b'|' => {
                i += 1;
                Tok::Pipe
            }
            b'(' => {
                i += 1;
                Tok::LParen
            }
            b')' => {
                i += 1;
                Tok::RParen
            }
            b'[' => {
                i += 1;
                Tok::LBracket
            }
            b']' => {
                i += 1;
                Tok::RBracket
            }
            b'{' => {
                i += 1;
                Tok::LBrace
            }
            b'}' => {
                i += 1;
                Tok::RBrace
            }
            b':' => {
                i += 1;
                Tok::Colon
            }
            b',' => {
                i += 1;
                Tok::Comma
            }
            b'=' => {
                if b.get(i + 1) == Some(&b'=') {
                    i += 2;
                    Tok::Op(CmpOp::Eq)
                } else {
                    return Err(QueryError::new(
                        src,
                        at,
                        "`=` is assignment in jq and is not supported; did you mean `==`?",
                    ));
                }
            }
            b'!' => {
                if b.get(i + 1) == Some(&b'=') {
                    i += 2;
                    Tok::Op(CmpOp::Ne)
                } else {
                    return Err(QueryError::new(src, at, "expected `!=`"));
                }
            }
            b'<' => {
                if b.get(i + 1) == Some(&b'=') {
                    i += 2;
                    Tok::Op(CmpOp::Le)
                } else {
                    i += 1;
                    Tok::Op(CmpOp::Lt)
                }
            }
            b'>' => {
                if b.get(i + 1) == Some(&b'=') {
                    i += 2;
                    Tok::Op(CmpOp::Ge)
                } else {
                    i += 1;
                    Tok::Op(CmpOp::Gt)
                }
            }
            b'"' => {
                let (s, next) = lex_string(src, i)?;
                i = next;
                Tok::Str(s)
            }
            b'-' | b'0'..=b'9' => {
                let (t, next) = lex_number(src, i)?;
                i = next;
                t
            }
            _ if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                Tok::Ident(src[start..i].to_string())
            }
            _ => {
                return Err(QueryError::new(
                    src,
                    at,
                    format!("unexpected character `{}`", c as char),
                ))
            }
        };
        out.push(Spanned { tok, at });
    }

    out.push(Spanned {
        tok: Tok::Eof,
        at: src.len(),
    });
    Ok(out)
}

/// Reads a JSON-style double-quoted string starting at `start`, returning the
/// unescaped contents and the offset just past the closing quote.
fn lex_string(src: &str, start: usize) -> QueryResult<(String, usize)> {
    let b = src.as_bytes();
    let mut i = start + 1;
    let mut s = String::new();
    while i < b.len() {
        match b[i] {
            b'"' => return Ok((s, i + 1)),
            b'\\' => {
                i += 1;
                let e = *b
                    .get(i)
                    .ok_or_else(|| QueryError::new(src, start, "unterminated string"))?;
                match e {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b't' => s.push('\t'),
                    b'r' => s.push('\r'),
                    b'b' => s.push('\u{8}'),
                    b'f' => s.push('\u{c}'),
                    b'u' => {
                        let hex = src
                            .get(i + 1..i + 5)
                            .ok_or_else(|| QueryError::new(src, i, "truncated `\\u` escape"))?;
                        let cp = u32::from_str_radix(hex, 16)
                            .map_err(|_| QueryError::new(src, i, "bad `\\u` escape"))?;
                        i += 4;
                        // Surrogate pair.
                        if (0xD800..0xDC00).contains(&cp) {
                            let ok = src.as_bytes().get(i + 1) == Some(&b'\\')
                                && src.as_bytes().get(i + 2) == Some(&b'u');
                            if ok {
                                let lo_hex = src.get(i + 3..i + 7).ok_or_else(|| {
                                    QueryError::new(src, i, "truncated surrogate pair")
                                })?;
                                let lo = u32::from_str_radix(lo_hex, 16)
                                    .map_err(|_| QueryError::new(src, i, "bad surrogate pair"))?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    i += 6;
                                    let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    s.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                } else {
                                    s.push('\u{FFFD}');
                                }
                            } else {
                                s.push('\u{FFFD}');
                            }
                        } else {
                            s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                    }
                    other => {
                        return Err(QueryError::new(
                            src,
                            i,
                            format!("unknown escape `\\{}`", other as char),
                        ))
                    }
                }
                i += 1;
            }
            _ => {
                // Copy the whole UTF-8 sequence.
                let ch = src[i..].chars().next().unwrap();
                s.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    Err(QueryError::new(src, start, "unterminated string"))
}

fn lex_number(src: &str, start: usize) -> QueryResult<(Tok, usize)> {
    let b = src.as_bytes();
    let mut i = start;
    if b[i] == b'-' {
        i += 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == int_start {
        return Err(QueryError::new(src, start, "expected a number"));
    }
    let mut is_float = false;
    if i < b.len() && b[i] == b'.' {
        is_float = true;
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        is_float = true;
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            // Without this the literal reaches `f64::parse`, fails, and
            // reports "out of range", which sends the reader looking for a
            // magnitude problem that is not there.
            return Err(QueryError::new(
                src,
                start,
                "number has no digits in its exponent",
            ));
        }
    }
    let text = &src[start..i];
    if !is_float {
        if let Ok(v) = text.parse::<i64>() {
            return Ok((Tok::Int(v), i));
        }
    }
    let v: f64 = text
        .parse()
        .map_err(|_| QueryError::new(src, start, "number literal out of range"))?;
    Ok((Tok::Num(v), i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::ir::CmpOp;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src)
            .unwrap_or_else(|e| panic!("{src}: {e}"))
            .into_iter()
            .map(|s| s.tok)
            .collect()
    }

    fn err(src: &str) -> String {
        lex(src).unwrap_err().message
    }

    #[test]
    fn lexes_every_punctuation_token() {
        assert_eq!(
            toks(".|()[]{}:,"),
            vec![
                Tok::Dot,
                Tok::Pipe,
                Tok::LParen,
                Tok::RParen,
                Tok::LBracket,
                Tok::RBracket,
                Tok::LBrace,
                Tok::RBrace,
                Tok::Colon,
                Tok::Comma,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn lexes_every_comparison_operator() {
        for (src, op) in [
            ("==", CmpOp::Eq),
            ("!=", CmpOp::Ne),
            ("<", CmpOp::Lt),
            ("<=", CmpOp::Le),
            (">", CmpOp::Gt),
            (">=", CmpOp::Ge),
        ] {
            assert_eq!(toks(src)[0], Tok::Op(op), "for `{src}`");
        }
    }

    #[test]
    fn distinguishes_integers_from_reals() {
        assert_eq!(toks("42")[0], Tok::Int(42));
        assert_eq!(toks("-42")[0], Tok::Int(-42));
        assert_eq!(toks("0")[0], Tok::Int(0));
        assert_eq!(toks("4.5")[0], Tok::Num(4.5));
        assert_eq!(toks("1e3")[0], Tok::Num(1000.0));
        assert_eq!(toks("1E3")[0], Tok::Num(1000.0));
        assert_eq!(toks("1e+3")[0], Tok::Num(1000.0));
        assert_eq!(toks("1e-3")[0], Tok::Num(0.001));
        assert_eq!(toks("-2.5E+10")[0], Tok::Num(-2.5e10));
        // Beyond i64, so it falls back to a float rather than failing.
        assert!(matches!(toks("99999999999999999999")[0], Tok::Num(_)));
    }

    #[test]
    fn rejects_malformed_numbers_with_a_specific_message() {
        assert!(err("-").contains("expected a number"));
        assert!(err("1e").contains("no digits in its exponent"));
        assert!(err("1e+").contains("no digits in its exponent"));
        assert!(err("1E-").contains("no digits in its exponent"));
    }

    #[test]
    fn a_trailing_decimal_point_is_accepted_as_jq_accepts_it() {
        // `1.` is not valid JSON, and the *data* scanner rejects it. In query
        // text it is harmless and jq allows it, so the lexer does too. Pinned
        // here so the asymmetry is deliberate rather than accidental.
        assert_eq!(toks("1.")[0], Tok::Num(1.0));
        assert!(crate::json::validate(br#"{"a":1.}"#).is_err());
    }

    #[test]
    fn decodes_string_escapes() {
        assert_eq!(toks(r#""plain""#)[0], Tok::Str("plain".into()));
        assert_eq!(toks(r#""a\"b""#)[0], Tok::Str("a\"b".into()));
        assert_eq!(toks(r#""a\\b""#)[0], Tok::Str("a\\b".into()));
        assert_eq!(toks(r#""a\/b""#)[0], Tok::Str("a/b".into()));
        assert_eq!(
            toks(r#""\b\f\n\r\t""#)[0],
            Tok::Str("\u{8}\u{c}\n\r\t".into())
        );
        assert_eq!(toks(r#""A""#)[0], Tok::Str("A".into()));
        assert_eq!(toks(r#""é""#)[0], Tok::Str("é".into()));
        assert_eq!(toks(r#""日""#)[0], Tok::Str("日".into()));
        // Surrogate pair for U+1F600.
        assert_eq!(toks(r#""😀""#)[0], Tok::Str("😀".into()));
        assert_eq!(toks(r#""""#)[0], Tok::Str(String::new()));
        // Literal UTF-8 passes through untouched.
        assert_eq!(toks("\"日本\"")[0], Tok::Str("日本".into()));
    }

    #[test]
    fn lone_surrogates_become_the_replacement_character() {
        // Matching the JSON scanner and the kernel, which both do this.
        assert_eq!(toks(r#""\ud83d""#)[0], Tok::Str("\u{FFFD}".into()));
        assert_eq!(toks(r#""\ud83dx""#)[0], Tok::Str("\u{FFFD}x".into()));
        assert_eq!(toks(r#""\ud83dA""#)[0], Tok::Str("\u{FFFD}A".into()));
    }

    #[test]
    fn rejects_malformed_strings() {
        assert!(err(r#""unterminated"#).contains("unterminated"));
        assert!(err(r#""bad \q escape""#).contains("unknown escape"));
        assert!(err(r#""\u00""#).contains("`\\u`"));
        assert!(err(r#""\uZZZZ""#).contains("`\\u`"));
        assert!(err(r#""\"#).contains("unterminated"));
    }

    #[test]
    fn lexes_identifiers_including_underscores_and_digits() {
        assert_eq!(toks("abc")[0], Tok::Ident("abc".into()));
        assert_eq!(toks("_a1")[0], Tok::Ident("_a1".into()));
        assert_eq!(toks("group_by")[0], Tok::Ident("group_by".into()));
    }

    #[test]
    fn skips_whitespace_and_comments() {
        assert_eq!(toks("  .  a  "), toks(".a"));
        assert_eq!(toks(".a # trailing comment"), toks(".a"));
        assert_eq!(toks("# leading\n.a"), toks(".a"));
        assert_eq!(toks("\t.a\r\n"), toks(".a"));
    }

    #[test]
    fn rejects_stray_characters_with_the_character_named() {
        assert!(err("@").contains("unexpected character `@`"));
        assert!(err("$x").contains("`$`"));
        assert!(err("&&").contains("`&`"));
    }

    #[test]
    fn assignment_is_called_out_specifically() {
        assert!(err(".a = 1").contains("did you mean `==`"));
        assert!(err("!").contains("expected `!=`"));
    }

    #[test]
    fn token_offsets_point_at_the_token_start() {
        let spanned = lex("select(.a == 1)").unwrap();
        let src = "select(.a == 1)";
        for s in &spanned {
            assert!(s.at <= src.len());
            assert!(src.is_char_boundary(s.at), "offset {} splits a char", s.at);
        }
        // The `==` is where the source says it is.
        let op = spanned
            .iter()
            .find(|s| matches!(s.tok, Tok::Op(_)))
            .unwrap();
        assert_eq!(&src[op.at..op.at + 2], "==");
    }

    #[test]
    fn offsets_are_char_boundaries_in_non_ascii_queries() {
        let src = r#"select(."日本" == "café")"#;
        for s in lex(src).unwrap() {
            assert!(
                src.is_char_boundary(s.at),
                "offset {} splits a multi-byte character",
                s.at
            );
        }
    }

    #[test]
    fn every_token_describes_itself_for_error_messages() {
        let all = [
            Tok::Dot,
            Tok::Pipe,
            Tok::LParen,
            Tok::RParen,
            Tok::LBracket,
            Tok::RBracket,
            Tok::LBrace,
            Tok::RBrace,
            Tok::Colon,
            Tok::Comma,
            Tok::Op(CmpOp::Eq),
            Tok::Ident("x".into()),
            Tok::Str("s".into()),
            Tok::Num(1.5),
            Tok::Int(2),
            Tok::Eof,
        ];
        for t in all {
            let d = t.describe();
            assert!(!d.is_empty(), "{t:?} describes itself as nothing");
        }
    }

    #[test]
    fn an_empty_query_lexes_to_just_eof() {
        assert_eq!(toks(""), vec![Tok::Eof]);
        assert_eq!(toks("   "), vec![Tok::Eof]);
    }
}
