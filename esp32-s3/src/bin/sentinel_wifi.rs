#![no_std]
#![no_main]
//! Tiered Edge AI — Sentinel with live WiFi wake to UNO Q.
//!
//! Uses framed protocol: 4-byte magic "TEAI" + 4-byte little-endian payload length
//! + FeatureTransfer payload. Server replies with 4-byte JSON length + JSON.

use defmt::{info, warn};
use embassy_executor::Spawner;
use embassy_net::{Runner, Stack, StackResources, tcp::TcpSocket};
use embassy_time::{Duration, Timer};
use embedded_io_async::{Read, Write};
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock, interrupt::software::SoftwareInterruptControl, rng::Rng,
    timer::timg::TimerGroup,
};
use esp_println as _;
use esp_radio::wifi::{
    Config, ControllerConfig, Interface, WifiController, sta::StationConfig,
};
use microflow::model;
use nalgebra::matrix;

extern crate alloc;

esp_bootloader_esp_idf::esp_app_desc!();

mod tiered_config {
    include!(concat!(env!("OUT_DIR"), "/tiered_config.rs"));
}

macro_rules! mk_static {
    ($t:ty, $val:expr) => {{
        static STATIC_CELL: static_cell::StaticCell<$t> = static_cell::StaticCell::new();
        STATIC_CELL.uninit().write($val)
    }};
}

const WIFI_SSID: &str = env!("TIERED_WIFI_SSID");
const WIFI_PASSWORD: &str = env!("TIERED_WIFI_PASSWORD");

const UNO_Q_IP: [u8; 4] = tiered_config::UNO_Q_IP;
const UNO_Q_PORT: u16 = tiered_config::UNO_Q_PORT;
const CONFIDENCE_THRESHOLD: f32 = 0.75;
const INFERENCE_INTERVAL_MS: u64 = 5000;

#[model("models/sine.tflite")]
struct Sentinel;

fn assess_confidence(prediction: f32) -> f32 {
    if prediction >= 0.0 && prediction <= 1.0 {
        1.0 - (prediction - 0.5).abs()
    } else {
        0.0
    }
}

fn build_wake_packet(prediction: f32, confidence: f32, feature: f32) -> heapless::Vec<u8, 64> {
    let mut payload = heapless::Vec::<u8, 56>::new();
    payload.extend_from_slice(b"TEAI").unwrap();
    payload.extend_from_slice(&1u32.to_le_bytes()).unwrap();
    payload.extend_from_slice(&prediction.to_le_bytes()).unwrap();
    payload.extend_from_slice(&confidence.to_le_bytes()).unwrap();
    payload.extend_from_slice(&1u32.to_le_bytes()).unwrap();
    payload.extend_from_slice(&feature.to_le_bytes()).unwrap();
    payload.extend_from_slice(&0u64.to_le_bytes()).unwrap();

    let mut pkt = heapless::Vec::<u8, 64>::new();
    pkt.extend_from_slice(b"TEAI").unwrap();
    pkt.extend_from_slice(&(payload.len() as u32).to_le_bytes()).unwrap();
    pkt.extend_from_slice(&payload).unwrap();
    pkt
}

#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(size: 64 * 1024);
    esp_alloc::heap_allocator!(size: 36 * 1024);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_int = SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_int.software_interrupt0);

    info!("[sentinel-wifi] boot");
    if WIFI_SSID.is_empty() || WIFI_PASSWORD.is_empty() {
        warn!("[sentinel-wifi] missing WiFi config; set TIERED_WIFI_SSID and TIERED_WIFI_PASSWORD before building");
    }

    let station_config = Config::Station(
        StationConfig::default()
            .with_ssid(WIFI_SSID)
            .with_password(WIFI_PASSWORD.into()),
    );

    let (controller, interfaces) = esp_radio::wifi::new(
        peripherals.WIFI,
        ControllerConfig::default().with_initial_config(station_config),
    )
    .expect("failed to initialize WiFi controller");

    let wifi_interface = interfaces.station;
    let net_config = embassy_net::Config::dhcpv4(Default::default());
    let rng = Rng::new();
    let seed = (rng.random() as u64) << 32 | rng.random() as u64;

    let (stack, runner) = embassy_net::new(
        wifi_interface,
        net_config,
        mk_static!(StackResources<3>, StackResources::<3>::new()),
        seed,
    );

    spawner.spawn(connection_task(controller).unwrap());
    spawner.spawn(net_task(runner).unwrap());

    info!("[sentinel-wifi] waiting for DHCP...");
    loop {
        stack.wait_config_up().await;
        if let Some(cfg) = stack.config_v4() {
            info!("[sentinel-wifi] DHCP up: ip={}", cfg.address);
            break;
        }
        Timer::after(Duration::from_millis(500)).await;
    }

    Timer::after(Duration::from_secs(2)).await;

    let mut cycle: u32 = 0;
    loop {
        let sensor_input = if cycle % 5 == 0 { 2.5f32 } else { (cycle as f32 * 0.1) % 6.28 };

        let prediction = Sentinel::predict(matrix![sensor_input])[0];
        let confidence = assess_confidence(prediction);

        info!("[sentinel-wifi] cycle={} input={} pred={} conf={}", cycle, sensor_input, prediction, confidence);

        if confidence < CONFIDENCE_THRESHOLD {
            warn!("[sentinel-wifi] LOW CONFIDENCE ({}); waking UNO Q", confidence);
            send_wake(stack, prediction, confidence, sensor_input).await;
        }

        cycle += 1;
        Timer::after(Duration::from_millis(INFERENCE_INTERVAL_MS)).await;
    }
}

async fn send_wake(stack: Stack<'static>, prediction: f32, confidence: f32, feature: f32) {
    let pkt = build_wake_packet(prediction, confidence, feature);

    let mut rx_buffer = [0u8; 512];
    let mut tx_buffer = [0u8; 512];
    let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
    socket.set_timeout(Some(Duration::from_secs(8)));

    let addr = embassy_net::IpAddress::v4(UNO_Q_IP[0], UNO_Q_IP[1], UNO_Q_IP[2], UNO_Q_IP[3]);

    match socket.connect((addr, UNO_Q_PORT)).await {
        Ok(()) => {
            info!("[sentinel-wifi] connected to UNO Q");
            match socket.write_all(&pkt).await {
                Ok(()) => {
                    let _ = socket.flush().await;
                    info!("[sentinel-wifi] wake packet sent ({} bytes)", pkt.len());

                    let mut len_buf = [0u8; 4];
                    match socket.read(&mut len_buf).await {
                        Ok(4) => {
                            let resp_len = u32::from_le_bytes(len_buf) as usize;
                            if resp_len > 0 && resp_len <= 256 {
                                let mut buf = [0u8; 256];
                                match socket.read_exact(&mut buf[..resp_len]).await {
                                    Ok(()) => {
                                        let text = core::str::from_utf8(&buf[..resp_len]).unwrap_or("<bad utf8>");
                                        info!("[sentinel-wifi] UNO Q response ({} bytes): {}", resp_len, text);
                                    }
                                    Err(e) => warn!("[sentinel-wifi] read_exact error: {:?}", e),
                                }
                            } else {
                                warn!("[sentinel-wifi] bad response length: {}", resp_len);
                            }
                        }
                        Ok(n) => warn!("[sentinel-wifi] short length header: {}", n),
                        Err(e) => warn!("[sentinel-wifi] read length error: {:?}", e),
                    }
                }
                Err(e) => warn!("[sentinel-wifi] write error: {:?}", e),
            }
        }
        Err(e) => warn!("[sentinel-wifi] connect to UNO Q failed: {:?}", e),
    }
}

#[embassy_executor::task]
async fn connection_task(mut controller: WifiController<'static>) {
    loop {
        info!("[wifi] connect attempt");
        match controller.connect_async().await {
            Ok(info) => {
                info!("[wifi] connected: ssid={} channel={} auth={:?}", info.ssid.as_str(), info.channel, info.authmode);
                let _ = controller.wait_for_disconnect_async().await;
                warn!("[wifi] disconnected");
            }
            Err(e) => {
                warn!("[wifi] connect failed: {:?}", e);
                Timer::after(Duration::from_secs(5)).await;
            }
        }
    }
}

#[embassy_executor::task]
async fn net_task(mut runner: Runner<'static, Interface<'static>>) {
    runner.run().await
}
