#![no_std]
#![no_main]

use defmt::{error, info, warn};
use embassy_executor::Spawner;
use embassy_net::{Runner, StackResources};
use embassy_time::{Duration, Timer};
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock, interrupt::software::SoftwareInterruptControl, ram, rng::Rng,
    timer::timg::TimerGroup,
};
use esp_println as _;
use esp_radio::wifi::{
    Config, ControllerConfig, Interface, WifiController, scan::ScanConfig, sta::StationConfig,
};

extern crate alloc;

esp_bootloader_esp_idf::esp_app_desc!();

macro_rules! mk_static {
    ($t:ty, $val:expr) => {{
        static STATIC_CELL: static_cell::StaticCell<$t> = static_cell::StaticCell::new();
        STATIC_CELL.uninit().write($val)
    }};
}

const WIFI_SSID: &str = env!("TIERED_WIFI_SSID");
const WIFI_PASSWORD_PRIMARY: &str = env!("TIERED_WIFI_PASSWORD");
const WIFI_PASSWORD_FALLBACK: &str = env!("TIERED_WIFI_PASSWORD_FALLBACK");

#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // WiFi needs substantially more heap than pure compute smoke tests.
    esp_alloc::heap_allocator!(#[ram(reclaimed)] size: 64 * 1024);
    esp_alloc::heap_allocator!(size: 36 * 1024);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_int = SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_int.software_interrupt0);

    info!("wifi-connect boot");
    if WIFI_SSID.is_empty() || WIFI_PASSWORD_PRIMARY.is_empty() {
        warn!("missing WiFi config; set TIERED_WIFI_SSID and TIERED_WIFI_PASSWORD before building");
    }

    let station_config = station_config(WIFI_PASSWORD_PRIMARY);
    let (mut controller, interfaces) = esp_radio::wifi::new(
        peripherals.WIFI,
        ControllerConfig::default().with_initial_config(station_config),
    )
    .expect("failed to initialize WiFi controller");

    let scan_config = ScanConfig::default().with_max(16);
    match controller.scan_async(&scan_config).await {
        Ok(aps) => {
            let mut found = false;
            for ap in aps {
                if ap.ssid.as_str() == WIFI_SSID {
                    found = true;
                    info!(
                        "target AP visible: ssid={} channel={} rssi={}",
                        ap.ssid.as_str(),
                        ap.channel,
                        ap.signal_strength
                    );
                }
            }
            if !found {
                warn!("target AP not visible in scan: ssid={}", WIFI_SSID);
            }
        }
        Err(e) => warn!("wifi scan failed: {:?}", e),
    }

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

    loop {
        stack.wait_config_up().await;
        if let Some(config) = stack.config_v4() {
            info!("wifi DHCP up: ip={}", config.address);
        } else {
            warn!("wifi stack reported up without ipv4 config");
        }

        Timer::after(Duration::from_secs(10)).await;
    }
}

fn station_config(password: &str) -> Config {
    Config::Station(
        StationConfig::default()
            .with_ssid(WIFI_SSID)
            .with_password(password.into()),
    )
}

#[embassy_executor::task]
async fn connection_task(mut controller: WifiController<'static>) {
    loop {
        info!("wifi connect attempt: primary credential");
        controller
            .set_config(&station_config(WIFI_PASSWORD_PRIMARY))
            .expect("failed to configure primary WiFi credential");

        match controller.connect_async().await {
            Ok(info) => {
                info!(
                    "wifi connected: ssid={} channel={} auth={}",
                    info.ssid.as_str(),
                    info.channel,
                    info.authmode
                );
                if let Ok(rssi) = controller.rssi() {
                    info!("wifi rssi={}dBm", rssi);
                }
                let disconnect = controller.wait_for_disconnect_async().await.ok();
                warn!("wifi disconnected: {:?}", disconnect);
            }
            Err(e) => {
                warn!("primary credential failed: {:?}", e);
                Timer::after(Duration::from_secs(2)).await;

                info!("wifi connect attempt: fallback credential");
                controller
                    .set_config(&station_config(WIFI_PASSWORD_FALLBACK))
                    .expect("failed to configure fallback WiFi credential");

                match controller.connect_async().await {
                    Ok(info) => {
                        info!(
                            "wifi connected: ssid={} channel={} auth={}",
                            info.ssid.as_str(),
                            info.channel,
                            info.authmode
                        );
                        if let Ok(rssi) = controller.rssi() {
                            info!("wifi rssi={}dBm", rssi);
                        }
                        let disconnect = controller.wait_for_disconnect_async().await.ok();
                        warn!("wifi disconnected: {:?}", disconnect);
                    }
                    Err(e) => error!("fallback credential failed: {:?}", e),
                }
            }
        }

        Timer::after(Duration::from_secs(5)).await;
    }
}

#[embassy_executor::task]
async fn net_task(mut runner: Runner<'static, Interface<'static>>) {
    runner.run().await
}
