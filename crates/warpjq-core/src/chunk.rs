//! Input handling: mmap or streamed reads, split into newline-aligned chunks.
//!
//! Chunk boundaries never land mid-line. The chunker walks back from the
//! nominal end to the last `\n` and hands the remainder to the next chunk, so
//! every consumer, CPU or GPU alike, sees whole lines and nothing else. That
//! invariant is what lets the GPU kernel assume "one line per warp" without
//! any cross-chunk stitching logic.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use memmap2::Mmap;

/// Default chunk size. Large enough to amortise a PCIe transfer, small enough
/// that three of them (in-flight, computing, draining) fit comfortably beside
/// the result buffers on an 8 GB card.
pub const DEFAULT_CHUNK_BYTES: usize = 256 << 20;

/// Ceiling on a single line, so a file of one enormous line cannot make us
/// allocate without bound. Overridable from the CLI.
pub const DEFAULT_MAX_LINE_BYTES: usize = 64 << 20;

/// A newline-aligned run of complete lines.
pub struct Chunk<'a> {
    pub data: &'a [u8],
    /// 1-based line number of the first line in this chunk, for diagnostics.
    pub first_line: u64,
}

/// Where the bytes come from.
pub enum Input {
    /// A regular file, mapped. Chunks borrow directly from the mapping, so
    /// there is no copy before the GPU staging buffer.
    Mapped { map: Mmap, name: String },
    /// stdin or anything else non-seekable, read into a growable buffer.
    Streamed {
        reader: Box<dyn Read + Send>,
        name: String,
    },
    /// Several inputs presented as one continuous stream.
    ///
    /// This exists because `warpjq 'sum(.n)' a.ndjson b.ndjson` has to total
    /// across both files. Running each file as its own query and letting the
    /// aggregate finish per file emits one row per file, `3` and `3` instead
    /// of `6`, and for `group_by` it emits duplicate rows for any key present
    /// in more than one file. jq treats a file list as one stream; so does this.
    ///
    /// Chunks never span a file boundary, which is what keeps it correct: a
    /// file's last line is complete at EOF whether or not it ends in a newline.
    Chained { inputs: Vec<Input>, name: String },
}

impl Input {
    pub fn open(path: &Path) -> io::Result<Input> {
        let file = File::open(path)?;
        let name = path.display().to_string();
        let meta = file.metadata()?;
        // Mapping an empty file fails on some platforms, and mapping a pipe
        // or device that reports a size is a trap. Only map regular files.
        if meta.is_file() && meta.len() > 0 {
            // SAFETY: we only read the mapping, and we accept the documented
            // risk that another process truncating the file underneath us can
            // fault, the same risk every mmap-based CLI takes.
            let map = unsafe { Mmap::map(&file)? };
            #[cfg(unix)]
            {
                // We stream front-to-back exactly once.
                let _ = map.advise(memmap2::Advice::Sequential);
            }
            return Ok(Input::Mapped { map, name });
        }
        Ok(Input::Streamed {
            reader: Box::new(file),
            name,
        })
    }

    /// An in-memory input, for tests and for `--bench`.
    pub fn from_bytes(data: &[u8]) -> Input {
        Input::Streamed {
            reader: Box::new(io::Cursor::new(data.to_vec())),
            name: "<memory>".to_string(),
        }
    }

    pub fn stdin() -> Input {
        Input::Streamed {
            reader: Box::new(io::stdin()),
            name: "<stdin>".to_string(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Input::Mapped { name, .. }
            | Input::Streamed { name, .. }
            | Input::Chained { name, .. } => name,
        }
    }

    /// Presents several inputs as one stream. A single input is returned
    /// unwrapped so the common case carries no indirection.
    pub fn chain(mut inputs: Vec<Input>) -> Input {
        if inputs.len() == 1 {
            return inputs.pop().unwrap();
        }
        let name = match inputs.len() {
            0 => "<empty>".to_string(),
            n => format!("{} (+{} more)", inputs[0].name(), n - 1),
        };
        Input::Chained { inputs, name }
    }

    /// Total size when known. `None` for streams, which disables the progress
    /// estimate rather than guessing.
    pub fn len_hint(&self) -> Option<u64> {
        match self {
            Input::Mapped { map, .. } => Some(map.len() as u64),
            Input::Streamed { .. } => None,
            // Known only if every part is.
            Input::Chained { inputs, .. } => {
                inputs.iter().map(|i| i.len_hint()).sum::<Option<u64>>()
            }
        }
    }

    /// Calls `f` with successive newline-aligned chunks.
    ///
    /// Chunks are yielded in file order and never overlap; concatenating them
    /// reproduces the input exactly (modulo a stripped BOM).
    ///
    /// `f` returns how many physical lines the chunk held, which is how
    /// `Chunk::first_line` advances. That looks like an odd thing to push onto
    /// the caller until you measure it: counting newlines here cost more than
    /// every GPU kernel combined, because it is a single-threaded pass over
    /// every byte *and* it is the first touch of each mapped page, so it eats
    /// the page faults too. Both backends already know the count as a
    /// by-product of the work they were going to do anyway. A caller that does
    /// not track lines returns 0, and `first_line` becomes a lower bound used
    /// only for diagnostics.
    pub fn for_each_chunk<F>(
        &mut self,
        chunk_bytes: usize,
        max_line_bytes: usize,
        mut f: F,
    ) -> io::Result<()>
    where
        F: FnMut(Chunk<'_>) -> io::Result<u64>,
    {
        self.for_each_chunk_dyn(chunk_bytes, max_line_bytes, &mut f)
    }

    /// The dynamically-dispatched body of [`Self::for_each_chunk`].
    ///
    /// `Chained` has to hand the same closure to each of its parts. Doing that
    /// generically would make the compiler instantiate `for_each_chunk::<&mut
    /// F>`, then `<&mut &mut F>`, and so on without a fixed point, since a
    /// `Chained` may itself contain one. A trait object ends the recursion.
    fn for_each_chunk_dyn(
        &mut self,
        chunk_bytes: usize,
        max_line_bytes: usize,
        f: &mut dyn FnMut(Chunk<'_>) -> io::Result<u64>,
    ) -> io::Result<()> {
        match self {
            Input::Mapped { map, .. } => {
                let mut chunker = Chunker::new(map, chunk_bytes, max_line_bytes);
                while let Some(c) = chunker.next_chunk()? {
                    let lines = f(c)?;
                    chunker.advance_lines(lines);
                }
                Ok(())
            }
            Input::Streamed { reader, .. } => stream_chunks(reader, chunk_bytes, max_line_bytes, f),
            Input::Chained { inputs, .. } => {
                for input in inputs.iter_mut() {
                    input.for_each_chunk_dyn(chunk_bytes, max_line_bytes, f)?;
                }
                Ok(())
            }
        }
    }
}

/// Strips a UTF-8 byte-order mark, which Windows tooling loves to prepend and
/// which would otherwise make the very first line fail to parse.
pub fn strip_bom(data: &[u8]) -> &[u8] {
    data.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(data)
}

/// Splits a borrowed buffer at newline boundaries.
pub struct Chunker<'a> {
    data: &'a [u8],
    pos: usize,
    target: usize,
    max_line: usize,
    line_no: u64,
}

impl<'a> Chunker<'a> {
    pub fn new(data: &'a [u8], target: usize, max_line: usize) -> Self {
        Chunker {
            data: strip_bom(data),
            pos: 0,
            target: target.max(1),
            max_line,
            line_no: 1,
        }
    }

    pub fn next_chunk(&mut self) -> io::Result<Option<Chunk<'a>>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let start = self.pos;
        let nominal_end = (start + self.target).min(self.data.len());

        let end = if nominal_end == self.data.len() {
            // Last chunk: take everything, newline-terminated or not.
            nominal_end
        } else {
            match memrchr(b'\n', &self.data[start..nominal_end]) {
                Some(rel) => start + rel + 1,
                None => {
                    // No newline in a whole chunk: a single line spans it.
                    // Extend forward to the next newline rather than splitting
                    // a line, but refuse to grow without bound.
                    match memchr(b'\n', &self.data[nominal_end..]) {
                        Some(rel) => nominal_end + rel + 1,
                        None => self.data.len(),
                    }
                }
            }
        };

        if end - start > self.max_line && !self.chunk_has_newline(start, end) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "line {} is at least {} bytes, over the {} byte limit \
                     (raise it with --max-line-bytes)",
                    self.line_no,
                    end - start,
                    self.max_line
                ),
            ));
        }

        let first_line = self.line_no;
        self.pos = end;
        Ok(Some(Chunk {
            data: &self.data[start..end],
            first_line,
        }))
    }

    /// Advances the line counter by what the consumer actually saw.
    pub fn advance_lines(&mut self, n: u64) {
        self.line_no += n;
    }

    fn chunk_has_newline(&self, start: usize, end: usize) -> bool {
        memchr(b'\n', &self.data[start..end.saturating_sub(1).max(start)]).is_some()
    }
}

/// Reads a non-seekable source in chunks, carrying the partial trailing line
/// forward into the next chunk.
fn stream_chunks(
    reader: &mut Box<dyn Read + Send>,
    chunk_bytes: usize,
    max_line_bytes: usize,
    f: &mut dyn FnMut(Chunk<'_>) -> io::Result<u64>,
) -> io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(chunk_bytes + (1 << 16));
    let mut carry: Vec<u8> = Vec::new();
    let mut line_no: u64 = 1;
    let mut first = true;

    loop {
        buf.clear();
        buf.extend_from_slice(&carry);
        carry.clear();

        let base = buf.len();
        // Read only up to the chunk size *including* whatever was carried
        // over, so a chunk never exceeds what the caller asked for. Getting
        // this wrong is not just untidy: the GPU backend sizes its staging
        // buffers from this number and hands anything larger to the CPU, so an
        // oversized chunk silently disabled the GPU for every streamed input.
        let want = if base >= chunk_bytes {
            // A single line longer than a chunk. Keep growing; the loop below
            // carries it forward until a newline turns up.
            chunk_bytes
        } else {
            chunk_bytes - base
        };
        buf.resize(base + want, 0);
        let mut filled = base;
        // `read` is allowed to return short; keep going until the buffer is
        // full or the source is done.
        loop {
            match reader.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => {
                    filled += n;
                    if filled == buf.len() {
                        break;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        buf.truncate(filled);

        if first {
            let n = buf.len();
            let stripped = strip_bom(&buf).len();
            if stripped != n {
                buf.drain(..n - stripped);
            }
            first = false;
        }

        if buf.is_empty() {
            return Ok(());
        }

        let eof = filled < base + want;
        let end = if eof {
            buf.len()
        } else {
            match memrchr(b'\n', &buf) {
                Some(rel) => rel + 1,
                None => {
                    if buf.len() > max_line_bytes {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "line {line_no} exceeds the {max_line_bytes} byte limit \
                                 (raise it with --max-line-bytes)"
                            ),
                        ));
                    }
                    // Nothing complete yet: carry it all and read more.
                    carry.extend_from_slice(&buf);
                    continue;
                }
            }
        };

        carry.extend_from_slice(&buf[end..]);
        let first_line = line_no;
        line_no += f(Chunk {
            data: &buf[..end],
            first_line,
        })?;

        if eof {
            return Ok(());
        }
    }
}

/// Iterates the logical lines of a chunk.
///
/// Empty and whitespace-only lines are skipped (the NDJSON spec permits blank
/// lines) and a trailing `\r` is trimmed so CRLF files behave.
pub struct Lines<'a> {
    data: &'a [u8],
    pos: usize,
    line_no: u64,
}

/// One line, with the number it had in the original file.
pub struct Line<'a> {
    pub bytes: &'a [u8],
    pub number: u64,
}

impl<'a> Lines<'a> {
    pub fn new(chunk: &Chunk<'a>) -> Self {
        Lines {
            data: chunk.data,
            pos: 0,
            line_no: chunk.first_line,
        }
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = Line<'a>;

    fn next(&mut self) -> Option<Line<'a>> {
        loop {
            if self.pos >= self.data.len() {
                return None;
            }
            let start = self.pos;
            let end = match memchr(b'\n', &self.data[start..]) {
                Some(rel) => start + rel,
                None => self.data.len(),
            };
            self.pos = end + 1;
            let number = self.line_no;
            self.line_no += 1;

            let mut line = &self.data[start..end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
            if line.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            return Some(Line {
                bytes: line,
                number,
            });
        }
    }
}

// Small, dependency-free byte searches. `memchr` the crate would be faster on
// huge scans, but these only run once per chunk boundary, not per line.
fn memchr(needle: u8, hay: &[u8]) -> Option<usize> {
    hay.iter().position(|&b| b == needle)
}

fn memrchr(needle: u8, hay: &[u8]) -> Option<usize> {
    hay.iter().rposition(|&b| b == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(data: &[u8], size: usize) -> Vec<(u64, String)> {
        let mut c = Chunker::new(data, size, DEFAULT_MAX_LINE_BYTES);
        let mut out = Vec::new();
        while let Some(ch) = c.next_chunk().unwrap() {
            let body = String::from_utf8(ch.data.to_vec()).unwrap();
            // Stand in for a real consumer, which is what advances the
            // chunker's line numbering.
            let lines = body.bytes().filter(|&b| b == b'\n').count() as u64;
            out.push((ch.first_line, body));
            c.advance_lines(lines);
        }
        out
    }

    #[test]
    fn chunks_always_end_on_a_newline() {
        let data = b"aaaa\nbb\ncccccc\ndd\n";
        for size in 1..=data.len() + 2 {
            let cs = chunks(data, size);
            let joined: String = cs.iter().map(|(_, s)| s.as_str()).collect();
            assert_eq!(joined.as_bytes(), data, "size={size}");
            for (i, (_, c)) in cs.iter().enumerate() {
                if i + 1 < cs.len() {
                    assert!(c.ends_with('\n'), "size={size} chunk={c:?}");
                }
            }
        }
    }

    #[test]
    fn line_numbers_are_continuous_across_chunks() {
        let data = b"1\n2\n3\n4\n5\n";
        let cs = chunks(data, 4);
        let mut expected = 1u64;
        for (first, body) in &cs {
            assert_eq!(*first, expected);
            expected += body.matches('\n').count() as u64;
        }
    }

    #[test]
    fn a_line_longer_than_a_chunk_is_not_split() {
        let long = "x".repeat(1000);
        let data = format!("a\n{long}\nb\n");
        let cs = chunks(data.as_bytes(), 16);
        assert!(cs.iter().any(|(_, c)| c.contains(&long)));
        let joined: String = cs.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(joined, data);
    }

    #[test]
    fn unterminated_last_line_is_still_yielded() {
        let cs = chunks(b"a\nb", 2);
        let joined: String = cs.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(joined, "a\nb");
    }

    #[test]
    fn bom_is_stripped() {
        let mut data = vec![0xEF, 0xBB, 0xBF];
        data.extend_from_slice(b"{}\n");
        let cs = chunks(&data, 64);
        assert_eq!(cs[0].1, "{}\n");
    }

    fn lines_of(data: &[u8]) -> Vec<(u64, String)> {
        let chunk = Chunk {
            data,
            first_line: 1,
        };
        Lines::new(&chunk)
            .map(|l| (l.number, String::from_utf8(l.bytes.to_vec()).unwrap()))
            .collect()
    }

    #[test]
    fn crlf_is_trimmed() {
        assert_eq!(
            lines_of(b"{\"a\":1}\r\n{\"b\":2}\r\n"),
            vec![(1, "{\"a\":1}".into()), (2, "{\"b\":2}".into())]
        );
    }

    #[test]
    fn blank_lines_are_skipped_but_still_counted() {
        // Line 3 keeps its real file line number even though line 2 is blank,
        // which is what makes error messages point at the right place.
        assert_eq!(
            lines_of(b"a\n\n   \nb\n"),
            vec![(1, "a".into()), (4, "b".into())]
        );
    }

    #[test]
    fn stream_and_mmap_paths_agree() {
        let data: Vec<u8> = (0..500)
            .map(|i| format!("{{\"i\":{i}}}\n"))
            .collect::<String>()
            .into_bytes();

        let mapped: Vec<String> = {
            let mut c = Chunker::new(&data, 97, DEFAULT_MAX_LINE_BYTES);
            let mut v = Vec::new();
            while let Some(ch) = c.next_chunk().unwrap() {
                for l in Lines::new(&ch) {
                    v.push(String::from_utf8(l.bytes.to_vec()).unwrap());
                }
            }
            v
        };

        let streamed: Vec<String> = {
            let mut r: Box<dyn Read + Send> = Box::new(io::Cursor::new(data.clone()));
            let mut v = Vec::new();
            stream_chunks(&mut r, 97, DEFAULT_MAX_LINE_BYTES, &mut |ch| {
                let mut n = 0;
                for l in Lines::new(&ch) {
                    v.push(String::from_utf8(l.bytes.to_vec()).unwrap());
                    n += 1;
                }
                Ok(n)
            })
            .unwrap();
            v
        };

        assert_eq!(mapped.len(), 500);
        assert_eq!(mapped, streamed);
    }

    /// A reader that hands back one byte per call, the way a slow pipe or
    /// socket can.
    struct DribbleReader(io::Cursor<Vec<u8>>);

    impl Read for DribbleReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            self.0.read(&mut buf[..1])
        }
    }

    #[test]
    fn a_bom_split_across_reads_is_still_stripped() {
        // The BOM check runs once, on the first chunk. That is only sound
        // because the fill loop keeps reading until the buffer is full or the
        // source is exhausted, so a 3-byte BOM can never be observed
        // half-delivered. This test pins that down with a reader that returns
        // a single byte per call.
        let mut data = vec![0xEF, 0xBB, 0xBF];
        data.extend_from_slice(b"{\"a\":1}\n{\"a\":2}\n");
        let mut r: Box<dyn Read + Send> = Box::new(DribbleReader(io::Cursor::new(data)));
        let mut lines = Vec::new();
        stream_chunks(&mut r, 8, DEFAULT_MAX_LINE_BYTES, &mut |ch| {
            let mut n = 0;
            for l in Lines::new(&ch) {
                lines.push(String::from_utf8(l.bytes.to_vec()).unwrap());
                n += 1;
            }
            Ok(n)
        })
        .unwrap();
        assert_eq!(
            lines,
            vec![r#"{"a":1}"#.to_string(), r#"{"a":2}"#.to_string()]
        );
    }

    #[test]
    fn stream_path_handles_a_line_spanning_several_reads() {
        let long = "y".repeat(5000);
        let data = format!("a\n{long}\nz\n").into_bytes();
        let mut r: Box<dyn Read + Send> = Box::new(io::Cursor::new(data));
        let mut lines = Vec::new();
        stream_chunks(&mut r, 64, DEFAULT_MAX_LINE_BYTES, &mut |ch| {
            let mut n = 0;
            for l in Lines::new(&ch) {
                lines.push(String::from_utf8(l.bytes.to_vec()).unwrap());
                n += 1;
            }
            Ok(n)
        })
        .unwrap();
        assert_eq!(lines, vec!["a".to_string(), long, "z".to_string()]);
    }
}
