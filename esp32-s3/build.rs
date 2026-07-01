fn main() {
    linker_be_nice();
    println!("cargo:rustc-link-arg=-Tdefmt.x");
    // make sure linkall.x is the last linker script (otherwise might cause problems with flip-link)
    println!("cargo:rustc-link-arg=-Tlinkall.x");
    emit_tiered_config();
}

fn emit_tiered_config() {
    println!("cargo:rerun-if-env-changed=TIERED_WIFI_SSID");
    println!("cargo:rerun-if-env-changed=TIERED_WIFI_PASSWORD");
    println!("cargo:rerun-if-env-changed=TIERED_WIFI_PASSWORD_FALLBACK");
    println!("cargo:rerun-if-env-changed=TIERED_UNO_Q_IP");
    println!("cargo:rerun-if-env-changed=TIERED_UNO_Q_PORT");

    let ssid = std::env::var("TIERED_WIFI_SSID").unwrap_or_default();
    let password = std::env::var("TIERED_WIFI_PASSWORD").unwrap_or_default();
    let fallback_password = std::env::var("TIERED_WIFI_PASSWORD_FALLBACK").unwrap_or_default();
    let ip = std::env::var("TIERED_UNO_Q_IP").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("TIERED_UNO_Q_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);

    println!("cargo:rustc-env=TIERED_WIFI_SSID={ssid}");
    println!("cargo:rustc-env=TIERED_WIFI_PASSWORD={password}");
    println!("cargo:rustc-env=TIERED_WIFI_PASSWORD_FALLBACK={fallback_password}");

    let octets = parse_ipv4(&ip).unwrap_or_else(|| panic!("invalid TIERED_UNO_Q_IP: {ip}"));
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let config_path = std::path::Path::new(&out_dir).join("tiered_config.rs");
    std::fs::write(
        config_path,
        format!(
            "pub const UNO_Q_IP: [u8; 4] = [{}, {}, {}, {}];\npub const UNO_Q_PORT: u16 = {};\n",
            octets[0], octets[1], octets[2], octets[3], port
        ),
    )
    .expect("failed to write generated tiered config");
}

fn parse_ipv4(ip: &str) -> Option<[u8; 4]> {
    let parts: Vec<_> = ip.split('.').collect();
    if parts.len() != 4 {
        return None;
    }

    let mut octets = [0u8; 4];
    for (idx, part) in parts.iter().enumerate() {
        octets[idx] = part.parse().ok()?;
    }
    Some(octets)
}

fn linker_be_nice() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let kind = &args[1];
        let what = &args[2];

        match kind.as_str() {
            "undefined-symbol" => match what.as_str() {
                what if what.starts_with("_defmt_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `defmt` not found - make sure `defmt.x` is added as a linker script and you have included `use defmt_rtt as _;`"
                    );
                    eprintln!();
                }
                "_stack_start" => {
                    eprintln!();
                    eprintln!("💡 Is the linker script `linkall.x` missing?");
                    eprintln!();
                }
                what if what.starts_with("esp_rtos_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `esp-radio` has no scheduler enabled. Make sure you have initialized `esp-rtos` or provided an external scheduler."
                    );
                    eprintln!();
                }
                "embedded_test_linker_file_not_added_to_rustflags" => {
                    eprintln!();
                    eprintln!(
                        "💡 `embedded-test` not found - make sure `embedded-test.x` is added as a linker script for tests"
                    );
                    eprintln!();
                }
                "free"
                | "malloc"
                | "calloc"
                | "get_free_internal_heap_size"
                | "malloc_internal"
                | "realloc_internal"
                | "calloc_internal"
                | "free_internal" => {
                    eprintln!();
                    eprintln!(
                        "💡 Did you forget the `esp-alloc` dependency or didn't enable the `compat` feature on it?"
                    );
                    eprintln!();
                }
                _ => (),
            },
            _ => {
                std::process::exit(1);
            }
        }

        std::process::exit(0);
    }

    println!(
        "cargo:rustc-link-arg=-Wl,--error-handling-script={}",
        std::env::current_exe().unwrap().display()
    );
}
