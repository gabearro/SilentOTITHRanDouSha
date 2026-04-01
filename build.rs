fn main() {
    println!("cargo:rerun-if-env-changed=SILENT_OT_ALLOW_DUMMY_METAL");
    #[cfg(target_os = "macos")]
    {
        compile_metal_shader("src/gpu/shader.metal", "beaver_triple");
        compile_metal_shader("src/gpu/shader32.metal", "beaver_triple32");
        compile_metal_shader("src/gpu/moe_shader.metal", "moe_swiglu");
    }
}

#[cfg(target_os = "macos")]
fn compile_metal_shader(src: &str, name: &str) {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let module_cache_dir = format!("{}/clang-module-cache", out_dir);
    let module_cache_arg = format!("-fmodules-cache-path={}", module_cache_dir);
    std::fs::create_dir_all(&module_cache_dir)
        .unwrap_or_else(|e| panic!("failed to create Metal module cache dir: {e}"));
    let allow_dummy = std::env::var("SILENT_OT_ALLOW_DUMMY_METAL")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);

    let air_path = format!("{}/{}.air", out_dir, name);
    let air_status = std::process::Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            &module_cache_arg,
            "-O2",
            "-std=metal3.0",
            "-c",
            src,
            "-o",
            &air_path,
        ])
        .status();

    match air_status {
        Ok(status) if status.success() => {}
        Ok(_) | Err(_) if allow_dummy => {
            eprintln!(
                "warning: Metal toolchain unavailable; using dummy metallib for {name}. \
Set SILENT_OT_ALLOW_DUMMY_METAL=0 (or unset) for strict builds."
            );
            std::fs::write(format!("{}/{}.metallib", out_dir, name), &[])
                .unwrap_or_else(|e| panic!("failed to write dummy metallib for {name}: {e}"));
            println!("cargo:rerun-if-changed={}", src);
            return;
        }
        Ok(_) => panic!("Metal shader compilation failed: {}", src),
        Err(e) => panic!("Failed to compile {} — is Xcode installed? {}", src, e),
    }

    let lib_path = format!("{}/{}.metallib", out_dir, name);
    let lib_status = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "-o", &lib_path, &air_path])
        .status();

    match lib_status {
        Ok(status) if status.success() => {}
        Ok(_) | Err(_) if allow_dummy => {
            eprintln!(
                "warning: Metal metallib link unavailable; using dummy metallib for {name}. \
Set SILENT_OT_ALLOW_DUMMY_METAL=0 (or unset) for strict builds."
            );
            std::fs::write(&lib_path, &[])
                .unwrap_or_else(|e| panic!("failed to write dummy metallib for {name}: {e}"));
        }
        Ok(_) => panic!("Metal library linking failed: {}", name),
        Err(e) => panic!("Failed to link {} metallib: {}", name, e),
    }

    println!("cargo:rerun-if-changed={}", src);
}
