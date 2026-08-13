//! The `warpjq` command line.

mod bench;
mod run;

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "warpjq",
    version,
    about = "jq for people with a GPU: filter, extract and aggregate gigabytes of NDJSON in seconds",
    long_about = None,
    args_conflicts_with_subcommands = true,
    after_help = EXAMPLES,
)]
struct Cli {
    #[command(flatten)]
    run: Option<RunArgs>,

    #[command(subcommand)]
    command: Option<Command>,
}

const EXAMPLES: &str = "\
Examples:
  warpjq 'select(.status == 500) | count'            access.ndjson
  warpjq 'select(.status >= 500)'                    access.ndjson
  warpjq '{t: .ts, path: .path, ms: .duration_ms}'   access.ndjson
  warpjq 'group_by(.host) | count'                   access.ndjson
  warpjq 'select(.status == 500) | sum(.bytes)'      access.ndjson

  warpjq gen --preset nginx --size 1GB -o access.ndjson
  warpjq bench 'select(.status == 500) | count' access.ndjson

Supported jq subset:
  paths          .a  .a.b  .a[0]  .\"odd key\"
  filters        select(.x == 1)  <  <=  >  >=  !=   and  or   | not
  projection     {k: .path, shorthand}
  aggregates     count  sum(.f)  min(.f)  max(.f)  avg(.f)
  grouping       group_by(.f) | <aggregate>

Anything outside that subset is rejected with a message saying so, rather than
being silently misread. See the README's Limitations section.";

#[derive(Args, Debug)]
struct RunArgs {
    /// The jq-subset query to run.
    query: String,

    /// Input files. Reads stdin when none are given.
    files: Vec<String>,

    /// Emit CSV instead of NDJSON.
    #[arg(long, conflicts_with = "count")]
    csv: bool,

    /// Print only how many rows the query produced.
    #[arg(long, short = 'c', conflicts_with = "csv")]
    count: bool,

    /// Which engine to use.
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    backend: BackendChoice,

    /// Abort on the first malformed line instead of skipping it.
    #[arg(long, conflicts_with = "skip_invalid")]
    strict: bool,

    /// Skip malformed lines without printing a warning.
    #[arg(long, conflicts_with = "strict")]
    skip_invalid: bool,

    /// CPU worker threads. 0 means one per core.
    #[arg(long, short = 'j', default_value_t = 0)]
    threads: usize,

    /// Bytes per pipeline chunk. Accepts suffixes: 256MB, 1GB.
    #[arg(long, value_name = "SIZE", default_value = "256MB")]
    chunk_size: String,

    /// Reject any single line longer than this.
    #[arg(long, value_name = "SIZE", default_value = "64MB")]
    max_line_bytes: String,

    /// Print a timing and throughput summary to stderr when finished.
    #[arg(long)]
    stats: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    /// GPU when one is usable, CPU otherwise. Never fails because of hardware.
    Auto,
    /// Force the GPU. Fails if it is unavailable, which is what you want in CI.
    Gpu,
    /// Force the CPU.
    Cpu,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate reproducible NDJSON test data.
    Gen(GenArgs),
    /// Time the same query on every available engine and print a table.
    Bench(BenchArgs),
}

#[derive(Args, Debug)]
struct GenArgs {
    /// Which shape of log to generate.
    #[arg(long, default_value = "nginx")]
    preset: String,

    /// How much to generate. Accepts suffixes: 500MB, 10GB.
    #[arg(long, default_value = "1GB")]
    size: String,

    /// Output file. Writes to stdout when omitted.
    #[arg(long, short = 'o')]
    output: Option<String>,

    /// Seed. The same seed always produces the same bytes.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// List the presets and exit.
    #[arg(long)]
    list: bool,
}

#[derive(Args, Debug)]
struct BenchArgs {
    /// The query to time.
    query: String,

    /// The file to run it over.
    file: String,

    /// Also time the system `jq`, if it is on PATH.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    jq: bool,

    /// Timed runs per engine. The reported figure is the best of these.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Untimed runs first, to warm the page cache.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match (cli.command, cli.run) {
        (Some(Command::Gen(args)), _) => run::generate(args),
        (Some(Command::Bench(args)), _) => bench::run(args),
        (None, Some(args)) => run::query(args),
        (None, None) => {
            // clap only reaches here when nothing at all was passed.
            eprintln!("warpjq: no query given\n\nTry 'warpjq --help'.");
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(code) => code,
        Err(e) => {
            // `warpjq ... | head` closes our stdout mid-stream. Every
            // well-behaved CLI exits quietly there; printing "the pipe has
            // been ended" at someone who just wanted the first ten rows is
            // noise, not information.
            if is_broken_pipe(&e) {
                return ExitCode::SUCCESS;
            }
            eprintln!("warpjq: {e}");
            // Distinguish "your query is wrong" from "something broke", so
            // scripts can tell a typo from a genuine failure.
            let code = if e.downcast_ref::<warpjq_core::error::QueryError>().is_some() {
                2
            } else {
                1
            };
            ExitCode::from(code)
        }
    }
}

/// True when the failure is just the reader going away.
///
/// Windows reports this as `ERROR_BROKEN_PIPE` (109) or `ERROR_NO_DATA` (232)
/// rather than the `BrokenPipe` kind, so the raw code is checked too.
fn is_broken_pipe(e: &anyhow::Error) -> bool {
    for cause in e.chain() {
        let io_err = cause
            .downcast_ref::<std::io::Error>()
            .or_else(|| match cause.downcast_ref::<warpjq_core::WarpError>() {
                Some(warpjq_core::WarpError::Io(io)) => Some(io),
                _ => None,
            });
        if let Some(io) = io_err {
            if io.kind() == std::io::ErrorKind::BrokenPipe {
                return true;
            }
            if matches!(io.raw_os_error(), Some(109) | Some(232)) {
                return true;
            }
        }
    }
    false
}

/// Parses `256MB`, `1GB`, `4096`, `1.5gb`.
pub fn parse_size(s: &str) -> anyhow::Result<u64> {
    let t = s.trim();
    let lower = t.to_ascii_lowercase();
    let (num, mult) = if let Some(v) = lower.strip_suffix("tb").or(lower.strip_suffix("t")) {
        (v, 1u64 << 40)
    } else if let Some(v) = lower.strip_suffix("gb").or(lower.strip_suffix("g")) {
        (v, 1u64 << 30)
    } else if let Some(v) = lower.strip_suffix("mb").or(lower.strip_suffix("m")) {
        (v, 1u64 << 20)
    } else if let Some(v) = lower.strip_suffix("kb").or(lower.strip_suffix("k")) {
        (v, 1u64 << 10)
    } else if let Some(v) = lower.strip_suffix("b") {
        (v, 1u64)
    } else {
        (lower.as_str(), 1u64)
    };
    let num: f64 = num
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("could not read `{s}` as a size (try 256MB or 1GB)"))?;
    // `f64::parse` happily accepts "nan", "inf" and "infinity". Casting those
    // to u64 saturates or zeroes silently, so --max-line-bytes inf would
    // become 0 and reject every line.
    if !num.is_finite() {
        anyhow::bail!("`{s}` is not a usable size (try 256MB or 1GB)");
    }
    if num < 0.0 {
        anyhow::bail!("size cannot be negative: `{s}`");
    }
    Ok((num * mult as f64) as u64)
}

/// Renders a byte count the way a human would say it.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else if v >= 100.0 {
        format!("{v:.0} {}", UNITS[i])
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn sizes_parse_with_and_without_suffixes() {
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("1KB").unwrap(), 1024);
        assert_eq!(parse_size("256MB").unwrap(), 256 << 20);
        assert_eq!(parse_size("1gb").unwrap(), 1 << 30);
        assert_eq!(parse_size("1.5GB").unwrap(), 1610612736);
        assert_eq!(parse_size(" 2 G ").unwrap(), 2 << 30);
        assert!(parse_size("banana").is_err());
        assert!(parse_size("-1GB").is_err());
    }

    #[test]
    fn byte_counts_render_readably() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(10 << 30), "10.0 GB");
    }

    #[test]
    fn a_bare_query_parses_as_a_run() {
        let cli = Cli::try_parse_from(["warpjq", "select(.a == 1)", "f.ndjson"]).unwrap();
        assert!(cli.command.is_none());
        let run = cli.run.unwrap();
        assert_eq!(run.query, "select(.a == 1)");
        assert_eq!(run.files, vec!["f.ndjson"]);
    }

    #[test]
    fn subcommands_still_win_over_the_positional_query() {
        let cli = Cli::try_parse_from(["warpjq", "gen", "--preset", "k8s"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Gen(_))));
    }

    #[test]
    fn csv_and_count_are_mutually_exclusive() {
        assert!(Cli::try_parse_from(["warpjq", ".a", "--csv", "--count"]).is_err());
    }

    #[test]
    fn strict_and_skip_invalid_are_mutually_exclusive() {
        assert!(Cli::try_parse_from(["warpjq", ".a", "--strict", "--skip-invalid"]).is_err());
    }
}
