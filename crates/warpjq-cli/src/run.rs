//! The `warpjq <query>` and `warpjq gen` entry points.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;

use anyhow::Context;
use warpjq_core::chunk::Input;
use warpjq_core::exec::{run_query, GpuStatus, OnInvalid, Options, Preference, RunStats};
use warpjq_core::gen::{self, Preset};
use warpjq_core::output::{Format, Writer};

use crate::{human_bytes, parse_size, BackendChoice, GenArgs, RunArgs};

pub fn query(args: RunArgs) -> anyhow::Result<ExitCode> {
    let program = warpjq_core::parse(&args.query)?;

    let options = Options {
        format: if args.csv {
            Format::Csv
        } else if args.count {
            Format::CountOnly
        } else {
            Format::Ndjson
        },
        chunk_bytes: parse_size(&args.chunk_size)? as usize,
        max_line_bytes: parse_size(&args.max_line_bytes)? as usize,
        on_invalid: if args.strict {
            OnInvalid::Abort
        } else if args.skip_invalid {
            OnInvalid::Skip
        } else {
            OnInvalid::Warn
        },
        threads: args.threads,
    };

    if options.chunk_bytes == 0 {
        anyhow::bail!("--chunk-size must be greater than zero");
    }

    let preference = match args.backend {
        BackendChoice::Auto => Preference::Auto,
        BackendChoice::Gpu => Preference::Gpu,
        BackendChoice::Cpu => Preference::Cpu,
    };

    // A GPU that was asked for but is missing should say so before we spend
    // time reading the file.
    if preference == Preference::Gpu {
        let status = GpuStatus::detect();
        if !status.is_available() {
            anyhow::bail!(
                "--backend gpu was requested but the GPU is not usable: {}",
                status.reason()
            );
        }
    }

    let stdout = io::stdout();
    let sink = BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut writer = Writer::new(sink, options.format);

    let mut inputs: Vec<Input> = Vec::new();
    if args.files.is_empty() {
        inputs.push(Input::stdin());
    } else {
        for f in &args.files {
            inputs
                .push(Input::open(Path::new(f)).with_context(|| format!("could not open `{f}`"))?);
        }
    }
    // One run over all the files, not one run per file. Running them
    // separately makes every aggregate finish per file: `sum(.n) a b` prints
    // one total per file instead of one total, and `group_by` emits duplicate
    // rows for any key that appears in more than one of them.
    let mut input = Input::chain(inputs);

    let started = std::time::Instant::now();
    let mut total = run_query(&program, &mut input, &options, &mut writer, preference)?;
    total.elapsed = started.elapsed();

    let (_, rows) = writer.finish()?;
    total.lines_out = rows;

    report(&total, &options, args.stats);

    // Exit 1 when nothing matched, like grep, so `if warpjq ...` works in
    // shell scripts.
    Ok(if rows == 0 && !program.is_aggregate() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

fn report(stats: &RunStats, options: &Options, verbose: bool) {
    if stats.malformed > 0 && options.on_invalid != OnInvalid::Skip {
        eprintln!(
            "warpjq: skipped {} malformed line{} (pass --strict to stop on the first one)",
            stats.malformed,
            if stats.malformed == 1 { "" } else { "s" }
        );
    }
    if stats.type_errors > 0 && options.on_invalid != OnInvalid::Skip {
        eprintln!(
            "warpjq: skipped {} line{} where a path hit a value of the wrong type \
             (jq reports these as \"cannot index\" errors)",
            stats.type_errors,
            if stats.type_errors == 1 { "" } else { "s" }
        );
    }
    if !verbose {
        return;
    }
    let secs = stats.elapsed.as_secs_f64();
    eprintln!(
        "warpjq: {backend} | {bytes} in {secs:.3}s = {gbps:.2} GB/s | \
         {lines_in} lines read, {lines_out} rows out",
        backend = stats.backend.map(|b| b.as_str()).unwrap_or("?"),
        bytes = human_bytes(stats.bytes_in),
        gbps = stats.throughput_gbps(),
        lines_in = stats.lines_in,
        lines_out = stats.lines_out,
    );
    if stats.gpu_fallback_lines > 0 {
        let pct = stats.gpu_fallback_lines as f64 / stats.lines_in.max(1) as f64 * 100.0;
        eprintln!(
            "warpjq: {} lines ({pct:.3}%) were finished on the CPU because the kernel \
             could not decide them exactly",
            stats.gpu_fallback_lines
        );
    }
}

pub fn generate(args: GenArgs) -> anyhow::Result<ExitCode> {
    if args.list {
        println!("Presets:");
        for p in Preset::all() {
            println!("  {:<12} {}", p.name(), p.describe());
            println!("  {:<12}   example query: {}", "", p.example_query());
        }
        return Ok(ExitCode::SUCCESS);
    }

    let preset = Preset::parse(&args.preset).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown preset `{}`; try one of: {}",
            args.preset,
            Preset::all()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let size = parse_size(&args.size)?;
    if size == 0 {
        anyhow::bail!("--size must be greater than zero");
    }

    let started = std::time::Instant::now();
    let stats = match &args.output {
        Some(path) => {
            let file = File::create(path).with_context(|| format!("could not create `{path}`"))?;
            let mut w = BufWriter::with_capacity(1 << 22, file);
            let s = gen::generate(&mut w, preset, size, args.seed)?;
            w.flush()?;
            s
        }
        None => {
            let stdout = io::stdout();
            let mut w = BufWriter::with_capacity(1 << 22, stdout.lock());
            gen::generate(&mut w, preset, size, args.seed)?
        }
    };

    if let Some(path) = args.output.as_ref() {
        let secs = started.elapsed().as_secs_f64();
        eprintln!(
            "warpjq: wrote {} ({} lines) of `{}` data with seed {} in {secs:.1}s",
            human_bytes(stats.bytes),
            stats.lines,
            preset.name(),
            args.seed
        );
        eprintln!("warpjq: try  warpjq '{}' {}", preset.example_query(), path);
    }
    Ok(ExitCode::SUCCESS)
}
