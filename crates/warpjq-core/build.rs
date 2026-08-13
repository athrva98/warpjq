//! Compiles the CUDA kernels into a static library and links it.
//!
//! This only runs when the `cuda` feature is on. A plain `cargo build` needs
//! no NVIDIA anything. That is deliberate, see README "Install".
//!
//! Env knobs:
//!   CUDA_PATH / CUDA_HOME  : toolkit root (auto-detected if unset)
//!   WARPJQ_CUDA_ARCH       : comma-separated SM versions, e.g. "86,120".
//!                             Defaults to a fat binary covering Volta..Blackwell.
//!   WARPJQ_NVCC_FLAGS      : extra flags appended verbatim.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CU_SOURCES: &[&str] = &["cuda/warpjq_kernels.cu"];

/// Architectures we ship in the release binary. The final entry is also
/// emitted as PTX so future GPUs JIT rather than fail to load.
const DEFAULT_ARCHS: &[&str] = &["75", "80", "86", "89", "90", "120"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WARPJQ_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=WARPJQ_NVCC_FLAGS");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    for src in CU_SOURCES {
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-changed=cuda/warpjq_kernels.h");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let cuda_root = find_cuda_root().unwrap_or_else(|| {
        panic!(
            "feature `cuda` is enabled but no CUDA toolkit was found. \
             Set CUDA_PATH, or build without --features cuda for the CPU-only binary."
        )
    });

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_name = if cfg!(target_os = "windows") {
        "warpjq_kernels.lib"
    } else {
        "libwarpjq_kernels.a"
    };
    let lib_path = out_dir.join(lib_name);

    let nvcc = cuda_root
        .join("bin")
        .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });

    let mut cmd = Command::new(&nvcc);
    cmd.arg("--lib").arg("-o").arg(&lib_path);

    let archs = arch_list(&nvcc);
    for arch in &archs {
        // Real SASS for each listed arch...
        cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    // ...plus PTX for the newest one, so unknown future GPUs still run.
    if let Some(newest) = archs.last() {
        cmd.arg(format!(
            "-gencode=arch=compute_{newest},code=compute_{newest}"
        ));
    }

    cmd.arg("-O3")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        // Line tables make `compute-sanitizer` and Nsight output readable
        // without the code-size hit of full -G device debug.
        .arg("-lineinfo");
    // Deliberately NOT --use_fast_math. There is no transcendental work in
    // these kernels to speed up, and it redefines INFINITY to a float and
    // relaxes NaN handling, both of which this code relies on being exact
    // (NaN is the "no numeric value here" sentinel in the aggregation path).

    if !cfg!(windows) {
        cmd.arg("-Xcompiler").arg("-fPIC");
    } else {
        // Match the CRT that Rust's MSVC target links against.
        cmd.arg("-Xcompiler").arg("/MD");
        // nvcc shells out to cl.exe and only looks on PATH, which means a
        // plain `cargo build` fails unless it was started from a Developer
        // Command Prompt. Finding the host compiler ourselves is the
        // difference between "clone and build" and a support issue.
        match find_msvc_bin() {
            Some(bin) => {
                cmd.arg("-ccbin").arg(&bin);
            }
            None => {
                if which("cl").is_none() {
                    panic!(
                        "could not find MSVC's cl.exe, which nvcc needs as its host \
                         compiler.\nInstall the \"Desktop development with C++\" \
                         workload for Visual Studio (the Build Tools are enough), or \
                         build from a Developer Command Prompt, or set \
                         WARPJQ_NVCC_FLAGS=\"-ccbin <path to the folder holding cl.exe>\"."
                    );
                }
            }
        }
    }

    if let Ok(extra) = env::var("WARPJQ_NVCC_FLAGS") {
        for f in extra.split_whitespace() {
            cmd.arg(f);
        }
    }

    for src in CU_SOURCES {
        cmd.arg(src);
    }

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed: {status}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=warpjq_kernels");

    for dir in cuda_link_dirs(&cuda_root) {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// Architectures this nvcc will actually accept, from `nvcc --list-gpu-arch`.
///
/// Returns an empty vec if the flag is unavailable, in which case the caller
/// does no filtering and lets nvcc decide.
fn supported_archs(nvcc: &Path) -> Vec<String> {
    let out = match Command::new(nvcc).arg("--list-gpu-arch").output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("compute_").map(str::to_string))
        .collect()
}

/// The architectures to compile for.
///
/// The default list spans several toolkit generations, and an older toolkit
/// rejects the newest entry outright: CUDA 12.6 fails with
/// `nvcc fatal: Unsupported gpu architecture 'compute_120'`, which made
/// `--features cuda` unbuildable on every toolkit before 12.8 rather than
/// simply building without Blackwell support. So the defaults are filtered to
/// what this nvcc knows.
///
/// An explicit `WARPJQ_CUDA_ARCH` is never filtered. Asking for an
/// architecture the toolkit cannot produce is an error worth reporting, not
/// one worth silently working around.
fn arch_list(nvcc: &Path) -> Vec<String> {
    if let Ok(s) = env::var("WARPJQ_CUDA_ARCH") {
        if !s.trim().is_empty() {
            let asked: Vec<String> = s
                .split(',')
                .map(|a| a.trim().trim_start_matches("sm_").to_string())
                .filter(|a| !a.is_empty())
                .collect();
            let supported = supported_archs(nvcc);
            if !supported.is_empty() {
                for a in &asked {
                    if !supported.contains(a) {
                        panic!(
                            "WARPJQ_CUDA_ARCH asks for sm_{a}, which this CUDA \
                             toolkit does not support.\nIt accepts: {}.",
                            supported.join(", ")
                        );
                    }
                }
            }
            return asked;
        }
    }

    let supported = supported_archs(nvcc);
    if supported.is_empty() {
        return DEFAULT_ARCHS.iter().map(|s| s.to_string()).collect();
    }
    let usable: Vec<String> = DEFAULT_ARCHS
        .iter()
        .filter(|a| supported.contains(&a.to_string()))
        .map(|s| s.to_string())
        .collect();
    if usable.is_empty() {
        panic!(
            "none of the default architectures ({}) are supported by this CUDA \
             toolkit, which accepts: {}.\nSet WARPJQ_CUDA_ARCH to one of those.",
            DEFAULT_ARCHS.join(", "),
            supported.join(", ")
        );
    }
    if usable.len() < DEFAULT_ARCHS.len() {
        let dropped: Vec<&str> = DEFAULT_ARCHS
            .iter()
            .filter(|a| !usable.contains(&a.to_string()))
            .copied()
            .collect();
        println!(
            "cargo:warning=CUDA toolkit does not support sm_{}; building for sm_{} only",
            dropped.join(", sm_"),
            usable.join(", sm_")
        );
    }
    usable
}

fn cuda_link_dirs(root: &Path) -> Vec<PathBuf> {
    if cfg!(windows) {
        vec![root.join("lib").join("x64")]
    } else {
        vec![root.join("lib64"), root.join("lib")]
    }
}

fn which(tool: &str) -> Option<PathBuf> {
    let cmd = if cfg!(windows) { "where" } else { "which" };
    let out = Command::new(cmd).arg(tool).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().map(|l| PathBuf::from(l.trim()))
}

/// Locates the directory holding a 64-bit `cl.exe`, preferring the newest
/// toolset of the newest Visual Studio install.
fn find_msvc_bin() -> Option<PathBuf> {
    if let Some(cl) = which("cl") {
        return cl.parent().map(|p| p.to_path_buf());
    }
    let program_files_x86 =
        env::var_os("ProgramFiles(x86)").unwrap_or_else(|| "C:\\Program Files (x86)".into());
    let vswhere = PathBuf::from(&program_files_x86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let out = Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    let root = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string());
    if root.as_os_str().is_empty() {
        return None;
    }
    let tools = root.join("VC").join("Tools").join("MSVC");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&tools)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    for v in versions.iter().rev() {
        let bin = v.join("bin").join("Hostx64").join("x64");
        if bin.join("cl.exe").exists() {
            return Some(bin);
        }
    }
    None
}

fn find_cuda_root() -> Option<PathBuf> {
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Some(p) = env::var_os(var) {
            let p = PathBuf::from(p);
            if p.join("bin").exists() {
                return Some(p);
            }
        }
    }
    // `which nvcc` -> strip /bin/nvcc
    let out = Command::new(if cfg!(windows) { "where" } else { "which" })
        .arg("nvcc")
        .output()
        .ok()?;
    if out.status.success() {
        let first = String::from_utf8_lossy(&out.stdout);
        let first = first.lines().next()?.trim();
        let p = Path::new(first).parent()?.parent()?;
        return Some(p.to_path_buf());
    }
    let fallback = PathBuf::from("/usr/local/cuda");
    fallback.join("bin").exists().then_some(fallback)
}
