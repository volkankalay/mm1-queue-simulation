"""M/M/1 kuyruk modeli - analitik hesaplama ve ayrık olay benzetimi.

Çalıştırma: python main.py
Grafikler için: pip install matplotlib
"""
import csv
import heapq
import random
from collections import deque
from dataclasses import dataclass
from statistics import mean, stdev

# Ödevde kullanılan sabitler
MU = 100.0
WARMUP_TIME = 100.0
MEASUREMENT_TIME = 5000.0
RHO_VALUES = [round(i * 0.05, 2) for i in range(2, 20)]
SEEDS = range(42, 52)
T_CRITICAL = 2.262157  # 10 tekrar için %95 güven aralığı (9 serbestlik derecesi)

# Ek analizlerde kullanılan değerler
ANALYSIS_RHO = 0.95
WARMUP_VALUES = [100.0, 500.0, 1000.0]
MEASUREMENT_TIMES = [1000.0, 5000.0, 10000.0]
WARMUP_ANALYSIS_TIME = 1000.0


# 1. Teorik hesaplamalar
@dataclass
class QueueMetrics:
    rho: float
    arrival_rate: float
    service_rate: float
    average_system_packets: float
    average_queue_packets: float
    average_system_time: float
    average_queue_time: float


def calculate_metrics(rho: float, mu: float = 100.0) -> QueueMetrics:
    """
    M/M/1 kuyruk performans metrikleri

    rho:            ρ       Trafik yoğunluğu (0 < rho < 1)
    mu:             μ       Servis orani (paket/saniye)
    arrival_rate:   λ       Gelis hizi
    """

    if not 0 < rho < 1:
        raise ValueError("M/M/1 modelinin kararlı olması için trafik yoğunluğunun 1'den küçük olması gerekir.")

    if mu <= 0:
        raise ValueError("μ değerinin pozitif olması gerekir.")

    """
    Geliş hızının hesaplanması;

    islem:  rho = lambda / mu 
    ise =>  arrival_rate = rho * mu
    """
    arrival_rate = rho * mu

    # Sistemdeki ortalama paket sayısı
    L = rho / (1 - rho)

    # Kuyruktaki ortalama paket sayısı
    Lq = (rho ** 2) / (1 - rho)

    # Sistemde ortalama kalış süresi
    W = 1 / (mu - arrival_rate)

    # Kuyrukta ortalama bekleme süresi
    Wq = arrival_rate / (mu * (mu - arrival_rate))

    # Little Yasası kontrolü
    # L = λ * W
    assert abs(L - arrival_rate * W) < 1e-10  # 1e-10 sistemsel tolerans icin eklendi
    assert abs(Lq - arrival_rate * Wq) < 1e-10

    return QueueMetrics(
        rho=rho,
        arrival_rate=arrival_rate,
        service_rate=mu,
        average_system_packets=L,
        average_queue_packets=Lq,
        average_system_time=W,
        average_queue_time=Wq
    )


# 2. Kuyruk simülasyonu
@dataclass
class SimulationResult:
    arrival_rate: float
    measured_arrival_rate: float
    service_rate: float
    rho: float
    total_packets: int
    simulation_time: float
    average_system_time: float
    average_queue_time: float
    average_system_packets: float
    average_queue_packets: float


def simulate_mm1(
        rho: float,
        mu: float = 100.0,
        warmup_time: float = 100.0,
        measurement_time: float = 1000.0,
        seed: int = 42
) -> SimulationResult:
    if not 0 < rho < 1:
        raise ValueError("Trafik yoğunluğu 0 ile 1 arasında olmalıdır.")

    if mu <= 0 or warmup_time < 0 or measurement_time <= 0:
        raise ValueError("Simülasyon parametreleri geçersiz.")

    rng = random.Random(seed)

    lam = rho * mu

    ARRIVAL = 0
    DEPARTURE = 1
    STOP = 2

    measurement_end = warmup_time + measurement_time

    event_list = []
    queue = deque()

    server_busy = False

    current_packet_arrival = None
    current_service_start = None
    current_packet_collect = False

    current_time = 0.0
    last_event_time = 0.0
    event_id = 0

    system_times = []
    queue_times = []

    area_system = 0.0
    area_queue = 0.0

    packets_in_system = 0

    # İlk paketin geliş zamanını belirle.
    first_arrival = rng.expovariate(lam)

    if first_arrival < measurement_end:
        heapq.heappush(
            event_list,
            (first_arrival, event_id, ARRIVAL)
        )
        event_id += 1

    # Ölçümün bitiş zamanını olay listesine ekle.
    heapq.heappush(
        event_list,
        (measurement_end, event_id, STOP)
    )
    event_id += 1

    # Olayları zaman sırasıyla işle.
    while event_list:

        event_time, _, event_type = heapq.heappop(event_list)

        # Yalnızca ölçüm aralığındaki süreyi hesaba kat.
        interval_start = max(last_event_time, warmup_time)
        interval_end = min(event_time, measurement_end)

        measured_interval = max(0.0, interval_end - interval_start)

        area_system += packets_in_system * measured_interval
        area_queue += len(queue) * measured_interval

        current_time = event_time
        last_event_time = current_time

        # Paket gelişi
        if event_type == ARRIVAL:

            packets_in_system += 1

            # Paket ölçüm dönemine ait mi?
            collect = warmup_time <= current_time < measurement_end

            # Sonraki paketin gelişini planla.
            next_arrival = current_time + rng.expovariate(lam)

            if next_arrival < measurement_end:
                heapq.heappush(
                    event_list,
                    (next_arrival, event_id, ARRIVAL)
                )
                event_id += 1

            if not server_busy:

                server_busy = True

                current_packet_arrival = current_time
                current_service_start = current_time
                current_packet_collect = collect

                service_time = rng.expovariate(mu)
                departure_time = current_time + service_time

                heapq.heappush(
                    event_list,
                    (departure_time, event_id, DEPARTURE)
                )
                event_id += 1

            else:

                queue.append((current_time, collect))

        # Paket çıkışı
        elif event_type == DEPARTURE:

            packets_in_system -= 1

            system_time = current_time - current_packet_arrival
            waiting_time = current_service_start - current_packet_arrival

            # Ölçüm döneminde gelen paketlerin sürelerini kaydet.
            if current_packet_collect:
                system_times.append(system_time)
                queue_times.append(waiting_time)

            if queue:

                next_packet_arrival, next_packet_collect = queue.popleft()

                current_packet_arrival = next_packet_arrival
                current_service_start = current_time
                current_packet_collect = next_packet_collect

                service_time = rng.expovariate(mu)
                departure_time = current_time + service_time

                heapq.heappush(
                    event_list,
                    (departure_time, event_id, DEPARTURE)
                )
                event_id += 1

            else:

                server_busy = False

                current_packet_arrival = None
                current_service_start = None
                current_packet_collect = False

        # Ölçüm sonu
        elif event_type == STOP:

            # Yeni geliş yok; kuyruktaki paketler tamamlanır.

            pass

    if not system_times:
        raise RuntimeError("Ölçüm süresinde paket kaydedilemedi.")

    average_system_time = sum(system_times) / len(system_times)
    average_queue_time = sum(queue_times) / len(queue_times)

    average_system_packets = area_system / measurement_time
    average_queue_packets = area_queue / measurement_time

    return SimulationResult(
        arrival_rate=lam,
        measured_arrival_rate=len(system_times) / measurement_time,
        service_rate=mu,
        rho=rho,
        total_packets=len(system_times),
        simulation_time=current_time,
        average_system_time=average_system_time,
        average_queue_time=average_queue_time,
        average_system_packets=average_system_packets,
        average_queue_packets=average_queue_packets
    )


# 3. Güven aralığı ve ana deneyler
def calculate_confidence_interval(values):
    sample_mean = mean(values)
    sample_std = stdev(values)

    sample_count = len(values)

    margin_of_error = (
            T_CRITICAL * sample_std / (sample_count ** 0.5)
    )

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return sample_mean, lower_bound, upper_bound


def run_experiments():
    results = []

    print("M/M/1 deneyleri başlıyor...\n")

    for rho in RHO_VALUES:

        print(f"Trafik yoğunluğu = {rho:.2f}")

        theoretical = calculate_metrics(rho, MU)

        simulated_w = []
        simulated_lq = []

        # Her yoğunluk için 10 bağımsız tekrar
        for seed in SEEDS:
            simulation = simulate_mm1(
                rho=rho,
                mu=MU,
                warmup_time=WARMUP_TIME,
                measurement_time=MEASUREMENT_TIME,
                seed=seed
            )

            simulated_w.append(
                simulation.average_system_time
            )

            simulated_lq.append(
                simulation.average_queue_packets
            )

        # %95 güven aralıkları
        w_mean, w_lower, w_upper = (
            calculate_confidence_interval(simulated_w)
        )

        lq_mean, lq_lower, lq_upper = (
            calculate_confidence_interval(simulated_lq)
        )

        result = {
            "rho": rho,
            "lambda": rho * MU,

            "theoretical_W": theoretical.average_system_time,
            "simulated_W": w_mean,
            "W_ci_lower": w_lower,
            "W_ci_upper": w_upper,

            "theoretical_Lq": theoretical.average_queue_packets,
            "simulated_Lq": lq_mean,
            "Lq_ci_lower": lq_lower,
            "Lq_ci_upper": lq_upper
        }

        results.append(result)

        print(
            f"  W:  {w_mean:.6f} "
            f"[{w_lower:.6f}, {w_upper:.6f}]"
        )

        print(
            f"  Lq: {lq_mean:.6f} "
            f"[{lq_lower:.6f}, {lq_upper:.6f}]\n"
        )

    # Sonuçları CSV dosyasına yaz
    with open(
            "experiment_results.csv",
            "w",
            newline="",
            encoding="utf-8"
    ) as file:

        fieldnames = results[0].keys()

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("Deneyler tamamlandı.")
    print("Sonuçlar experiment_results.csv dosyasına kaydedildi.")


# 4. Little Yasası kontrolü
def relative_error(simulated, theoretical):
    return abs(simulated - theoretical) / abs(theoretical) * 100


def validate_simulation(rho=0.50, mu=100.0):
    simulation = simulate_mm1(
        rho=rho,
        mu=mu,
        warmup_time=100.0,
        measurement_time=1000.0,
        seed=42
    )

    theoretical = calculate_metrics(rho, mu)

    # Measured arrival rate
    measured_lambda = simulation.measured_arrival_rate

    # Little's Law
    little_l = measured_lambda * simulation.average_system_time
    little_lq = measured_lambda * simulation.average_queue_time

    print("\n========== LITTLE YASASI KONTROLÜ ==========")

    print(f"Ölçülen geliş hızı: {measured_lambda:.6f}")

    print(
        f"L (simülasyon): {simulation.average_system_packets:.6f} "
        f"| Lambda * W: {little_l:.6f}"
    )

    print(
        f"Lq (simülasyon): {simulation.average_queue_packets:.6f} "
        f"| Lambda * Wq: {little_lq:.6f}"
    )

    print("\n========== BAĞIL HATALAR ==========")

    metrics = [
        (
            "W",
            simulation.average_system_time,
            theoretical.average_system_time
        ),
        (
            "Wq",
            simulation.average_queue_time,
            theoretical.average_queue_time
        ),
        (
            "L",
            simulation.average_system_packets,
            theoretical.average_system_packets
        ),
        (
            "Lq",
            simulation.average_queue_packets,
            theoretical.average_queue_packets
        )
    ]

    for name, simulated, expected in metrics:
        error = relative_error(simulated, expected)

        print(
            f"{name}: "
            f"Simülasyon = {simulated:.6f}, "
            f"Teorik = {expected:.6f}, "
            f"Hata = {error:.4f}%"
        )


def confidence_margin(values):
    """Tekrarların ortalamasını ve güven aralığının yarı genişliğini döndürür."""
    avg = mean(values)
    margin = T_CRITICAL * stdev(values) / (len(values) ** 0.5)
    return avg, margin


# 5. Isınma süresi analizi
def run_warmup_analysis():
    theoretical = calculate_metrics(ANALYSIS_RHO, MU)

    print("========== ISINMA SÜRESİ ANALİZİ ==========")

    print(f"Trafik yoğunluğu: {ANALYSIS_RHO}")
    print(f"Servis hızı: {MU}")
    print(f"Ölçüm süresi: {WARMUP_ANALYSIS_TIME} saniye")

    print(
        f"\nTeorik W: "
        f"{theoretical.average_system_time:.6f}"
    )

    print(
        f"Teorik Lq: "
        f"{theoretical.average_queue_packets:.6f}"
    )

    for warmup in WARMUP_VALUES:

        w_values = []
        lq_values = []

        for seed in SEEDS:
            result = simulate_mm1(
                rho=ANALYSIS_RHO,
                mu=MU,
                warmup_time=warmup,
                measurement_time=WARMUP_ANALYSIS_TIME,
                seed=seed
            )

            w_values.append(
                result.average_system_time
            )

            lq_values.append(
                result.average_queue_packets
            )

        w_mean, w_margin = confidence_margin(w_values)
        lq_mean, lq_margin = confidence_margin(lq_values)

        print(f"\n---------- Isınma süresi: {warmup:.0f} saniye ----------")

        print(
            f"W: {w_mean:.6f} "
            f"± {w_margin:.6f} saniye"
        )

        print(
            f"Lq: {lq_mean:.6f} "
            f"± {lq_margin:.6f} paket"
        )

        w_error = (
                abs(w_mean - theoretical.average_system_time)
                / theoretical.average_system_time
                * 100
        )

        lq_error = (
                abs(lq_mean - theoretical.average_queue_packets)
                / theoretical.average_queue_packets
                * 100
        )

        print(f"W bağıl hata: {w_error:.4f}%")
        print(f"Lq bağıl hata: {lq_error:.4f}%")


# 6. Ölçüm süresi analizi
def run_measurement_analysis():
    theoretical = calculate_metrics(ANALYSIS_RHO, MU)

    print("========== ÖLÇÜM SÜRESİ ANALİZİ ==========")
    print(f"Trafik yoğunluğu: {ANALYSIS_RHO}")
    print(f"Isınma süresi: {WARMUP_TIME} saniye")

    for measurement_time in MEASUREMENT_TIMES:

        w_values = []
        lq_values = []

        for seed in SEEDS:
            result = simulate_mm1(
                rho=ANALYSIS_RHO,
                mu=MU,
                warmup_time=WARMUP_TIME,
                measurement_time=measurement_time,
                seed=seed
            )

            w_values.append(result.average_system_time)
            lq_values.append(result.average_queue_packets)

        w_mean, w_margin = confidence_margin(w_values)
        lq_mean, lq_margin = confidence_margin(lq_values)

        w_error = (
                abs(w_mean - theoretical.average_system_time)
                / theoretical.average_system_time * 100
        )

        lq_error = (
                abs(lq_mean - theoretical.average_queue_packets)
                / theoretical.average_queue_packets * 100
        )

        print(f"\n--- Ölçüm süresi: {measurement_time:.0f} saniye ---")
        print(f"W:  {w_mean:.6f} ± {w_margin:.6f}")
        print(f"Lq: {lq_mean:.6f} ± {lq_margin:.6f}")
        print(f"W bağıl hata:  {w_error:.4f}%")
        print(f"Lq bağıl hata: {lq_error:.4f}%")


# 7. Sonuç grafikleri

def create_plots():
    """CSV sonuçlarından W ve Lq grafiklerini kaydeder."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Grafikler oluşturulamadı: matplotlib kurulu değil.")
        print("Kurulum: pip install matplotlib")
        return

    with open("experiment_results.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rho = [float(row["rho"]) for row in rows]
    graphs = [
        ("W", "theoretical_W", "simulated_W", "W_ci_lower", "W_ci_upper",
         "Ortalama sistemde kalma süresi (sn)", "rho_vs_W.png"),
        ("Lq", "theoretical_Lq", "simulated_Lq", "Lq_ci_lower", "Lq_ci_upper",
         "Ortalama kuyruk uzunluğu (paket)", "rho_vs_Lq.png"),
    ]

    for name, theoretical_key, simulated_key, lower_key, upper_key, ylabel, filename in graphs:
        theoretical = [float(row[theoretical_key]) for row in rows]
        simulated = [float(row[simulated_key]) for row in rows]
        lower = [float(row[lower_key]) for row in rows]
        upper = [float(row[upper_key]) for row in rows]
        errors = [
            [value - bound for value, bound in zip(simulated, lower)],
            [bound - value for value, bound in zip(simulated, upper)],
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(rho, theoretical, linewidth=2, label="Teorik M/M/1")
        plt.errorbar(rho, simulated, yerr=errors, fmt="o", capsize=4,
                     label="Simülasyon (%95 güven aralığı)")
        plt.xlabel("Trafik yoğunluğu (ρ)")
        plt.ylabel(ylabel)
        plt.title(f"Trafik yoğunluğu - {name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Grafik kaydedildi: {filename}")


# 8. Programın başlangıcı

def main():
    run_experiments()
    create_plots()
    validate_simulation()
    run_warmup_analysis()
    run_measurement_analysis()
    print("\nTüm işlemler tamamlandı.")


if __name__ == "__main__":
    main()
