import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = r"C:\Users\zulal\OneDrive\Masaüstü\RL_VİZE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# AKILLI SULAMA SİSTEMİ - Q LEARNING
# =====================================================
# Bu projede amaç:
# Toprak nemi, hava durumu, sıcaklık, rüzgar ve basınç bilgilerine göre
# sistemin sulama yapıp yapmayacağına karar vermesidir.
#
# Kullanılan yöntem:
# Q-Learning algoritması
# =====================================================


# =====================================================
# 1) STATE VE ACTION TANIMLARI
# =====================================================

# Toprak nem durumu
soil_states = ["Kuru", "Nemli", "Islak"]

# Hava durumu
weather_states = ["Yağmurlu", "Güneşli", "Bulutlu", "Karlı"]

# Hava sıcaklığı durumu
air_temp_states = ["Düşük", "Orta", "Yüksek"]

# Rüzgar durumu
wind_states = ["Düşük", "Orta", "Yüksek"]

# Basınç durumu
pressure_states = ["Düşük", "Normal", "Yüksek"]

# Sistemin alabileceği aksiyonlar
# 0 -> Sulama kapalı
# 1 -> Sulama açık
actions = ["Sulama kapalı", "Sulama açık"]


# =====================================================
# 2) Q-TABLE OLUŞTURMA
# =====================================================
# Q-table sistemin öğrenme hafızasıdır.
#
# Boyutlar:
# 3 -> Toprak durumu
# 4 -> Hava durumu
# 3 -> Sıcaklık durumu
# 3 -> Rüzgar durumu
# 3 -> Basınç durumu
# 2 -> Aksiyon sayısı
#
# Başlangıçta tüm Q değerleri sıfırdır.
# Eğitim ilerledikçe sistem doğru aksiyonların değerini yükseltir.

Q = np.zeros((3, 4, 3, 3, 3, 2))


# =====================================================
# 3) AYRIKLAŞTIRMA FONKSİYONLARI
# =====================================================
# Gerçek sensör verileri sürekli değerlerdir.
# Örneğin toprak nemi %64, sıcaklık 27°C olabilir.
# Q-Learning ise ayrık durumlarla çalıştığı için
# bu değerler düşük/orta/yüksek gibi sınıflara ayrılır.


def soil_moisture_to_state(moisture_pct):
    """
    Toprak nem yüzdesini ayrık duruma çevirir.

    %0 - %39  -> Kuru
    %40 - %75 -> Nemli
    %76 - %100 -> Islak
    """
    if moisture_pct < 40:
        return 0   # Kuru
    elif moisture_pct <= 75:
        return 1   # Nemli
    else:
        return 2   # Islak


def air_temperature_to_state(temp):
    """
    Hava sıcaklığını ayrık duruma çevirir.

    10°C ve altı -> Düşük
    11-25°C      -> Orta
    25°C üstü    -> Yüksek
    """
    if temp <= 10:
        return 0   # Düşük
    elif temp <= 25:
        return 1   # Orta
    else:
        return 2   # Yüksek


def wind_speed_to_state(wind_speed):
    """
    Rüzgar hızını ayrık duruma çevirir.

    10 km/h altı  -> Düşük
    10-25 km/h    -> Orta
    25 km/h üstü  -> Yüksek
    """
    if wind_speed < 10:
        return 0   # Düşük
    elif wind_speed <= 25:
        return 1   # Orta
    else:
        return 2   # Yüksek


def pressure_to_state(pressure):
    """
    Hava basıncını ayrık duruma çevirir.

    1000 hPa altı     -> Düşük
    1000-1020 hPa     -> Normal
    1020 hPa üstü     -> Yüksek
    """
    if pressure < 1000:
        return 0   # Düşük basınç
    elif pressure <= 1020:
        return 1   # Normal basınç
    else:
        return 2   # Yüksek basınç


# =====================================================
# 4) REWARD FONKSİYONU
# =====================================================
# Reward fonksiyonu sistemin kararını değerlendirir.
#
# Doğru karar verilirse pozitif ödül verilir.
# Yanlış karar verilirse negatif ceza verilir.
#
# Örnek:
# Toprak kuruysa ve sistem sulama yaparsa ödül alır.
# Toprak ıslakken sulama yaparsa ceza alır.


def get_reward(soil, weather, air_temp, wind, pressure, action):
    """
    Q-Learning için ödül fonksiyonu.

    action = 0 -> Sulama kapalı
    action = 1 -> Sulama açık
    """

    # Toprak kuruysa sulama yapmak doğru bir karardır.
    if soil == 0 and action == 1:
        reward = 20

        # Hava sıcak ve güneşliyse sulama daha gerekli olur.
        if air_temp == 2 and weather == 1:
            reward += 10

        # Yüksek basınç genellikle açık ve yağışsız hava anlamına gelir.
        if pressure == 2:
            reward += 5

        # Rüzgar çok yüksekse sulama verimi düşer.
        if wind == 2:
            reward -= 8

        return reward

    # Toprak kuruysa ve sulama yapılmazsa bitki zarar görür.
    if soil == 0 and action == 0:
        return -50

    # Toprak ıslakken sulama yapmak gereksizdir.
    if soil == 2 and action == 1:
        return -25

    # Toprak ıslakken sulama yapmamak doğru karardır.
    if soil == 2 and action == 0:
        return 15

    # Yağmurlu veya karlı havada sulama yapmak gereksizdir.
    if weather in [0, 3] and action == 1:
        return -15

    # Rüzgar çok yüksekken sulama yapmak verimsizdir.
    if wind == 2 and action == 1:
        return -10

    # Toprak nemliyse sulama yapmamak genelde doğrudur.
    if soil == 1 and action == 0:
        return 10

    # Toprak nemli ama hava sıcak ve güneşliyse az sulama kabul edilebilir.
    if soil == 1 and action == 1:
        if air_temp == 2 and weather == 1 and wind != 2:
            return 8
        return -5

    return 0


# =====================================================
# 5) ORTAM GEÇİŞ FONKSİYONU
# =====================================================
# Sistem bir aksiyon aldıktan sonra yeni durum oluşur.
#
# Sulama yapılırsa toprak daha nemli hale gelir.
# Sulama yapılmazsa sıcaklık, güneş ve rüzgar etkisiyle toprak kuruyabilir.


def next_state(soil, weather, air_temp, wind, pressure, action):
    """
    Aksiyon sonrası oluşacak yeni state bilgisini üretir.
    """

    # Sulama yapılırsa toprak nem seviyesi artar.
    if action == 1:
        soil = min(soil + 1, 2)

    # Sulama yapılmazsa hava şartlarına göre toprak kuruyabilir.
    else:
        # Güneşli ve sıcak havada toprak kurur.
        if weather == 1 and air_temp == 2:
            soil = max(soil - 1, 0)

        # Rüzgar yüksekse buharlaşma artar.
        if wind == 2:
            soil = max(soil - 1, 0)

    # Yeni hava koşulları rastgele oluşturulur.
    new_weather = random.randint(0, 3)
    new_air_temp = random.randint(0, 2)
    new_wind = random.randint(0, 2)
    new_pressure = random.randint(0, 2)

    return soil, new_weather, new_air_temp, new_wind, new_pressure


# =====================================================
# 6) Q-LEARNING EĞİTİMİ
# =====================================================

# Öğrenme oranı
alpha = 0.1

# Gelecekteki ödüllerin önem katsayısı
gamma = 0.9

# Keşif oranı
# Sistem bazı durumlarda rastgele karar vererek öğrenmeyi geliştirir.
epsilon = 0.2

# Eğitim bölüm sayısı
episodes = 5000

# Her bölümde alınan toplam reward değerleri burada tutulur.
episode_rewards = []


for episode in range(episodes):

    # Her eğitim bölümünde başlangıç state rastgele seçilir.
    soil = random.randint(0, 2)
    weather = random.randint(0, 3)
    air_temp = random.randint(0, 2)
    wind = random.randint(0, 2)
    pressure = random.randint(0, 2)

    total_episode_reward = 0

    # Her episode 30 adımdan oluşur.
    for step in range(30):

        # Epsilon-greedy seçim yöntemi
        # Bazen rastgele aksiyon seçilir.
        # Bazen Q-table'a göre en iyi aksiyon seçilir.
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[soil, weather, air_temp, wind, pressure])

        # Seçilen aksiyona göre ödül hesaplanır.
        reward = get_reward(soil, weather, air_temp, wind, pressure, action)
        total_episode_reward += reward

        # Yeni state hesaplanır.
        new_soil, new_weather, new_air_temp, new_wind, new_pressure = next_state(
            soil, weather, air_temp, wind, pressure, action
        )

        # Q-Learning güncelleme formülü
        Q[soil, weather, air_temp, wind, pressure, action] += alpha * (
            reward
            + gamma * np.max(Q[new_soil, new_weather, new_air_temp, new_wind, new_pressure])
            - Q[soil, weather, air_temp, wind, pressure, action]
        )

        # Yeni state mevcut state olur.
        soil = new_soil
        weather = new_weather
        air_temp = new_air_temp
        wind = new_wind
        pressure = new_pressure

    episode_rewards.append(total_episode_reward)

print("Q-Learning eğitimi tamamlandı.")


# =====================================================
# 7) 1 YILLIK SENSÖR VERİSİ ÜRETİMİ
# =====================================================
# Bu fonksiyon yılın gününe göre mevsimsel veri üretir.
# Örneğin yazın hava daha sıcak ve güneşli,
# kışın ise yağış ve kar ihtimali daha fazladır.


def generate_daily_sensor_data(day):
    """
    365 günlük örnek hava, sıcaklık, rüzgar ve basınç verisi üretir.
    """

    # Kış dönemi
    if day <= 90 or day >= 335:
        season_name = "Kış"
        air_temp_value = random.randint(-5, 15)
        wind_speed = random.randint(5, 35)
        pressure_value = random.randint(990, 1025)

        weather_name = random.choices(
            ["Güneşli", "Karlı", "Yağmurlu", "Bulutlu"],
            weights=[10, 20, 60, 10],
            k=1
        )[0]

        # Karlı havada sıcaklığın çok yüksek olmaması sağlanır.
        if weather_name == "Karlı" and air_temp_value > 3:
            air_temp_value = random.randint(-5, 3)

    # İlkbahar dönemi
    elif day <= 170:
        season_name = "İlkbahar"
        air_temp_value = random.randint(12, 25)
        wind_speed = random.randint(3, 25)
        pressure_value = random.randint(995, 1025)

        weather_name = random.choices(
            ["Güneşli", "Yağmurlu", "Bulutlu"],
            weights=[70, 10, 20],
            k=1
        )[0]

    # Yaz dönemi
    elif day <= 260:
        season_name = "Yaz"
        air_temp_value = random.randint(25, 40)
        wind_speed = random.randint(2, 30)
        pressure_value = random.randint(1005, 1030)

        weather_name = random.choices(
            ["Güneşli", "Yağmurlu", "Bulutlu"],
            weights=[85, 5, 10],
            k=1
        )[0]

    # Sonbahar dönemi
    else:
        season_name = "Sonbahar"
        air_temp_value = random.randint(10, 25)
        wind_speed = random.randint(5, 30)
        pressure_value = random.randint(990, 1025)

        weather_name = random.choices(
            ["Güneşli", "Yağmurlu", "Bulutlu"],
            weights=[40, 40, 10],
            k=1
        )[0]

    # Gerçek değerler ayrık state değerlerine çevrilir.
    weather = weather_states.index(weather_name)
    air_temp = air_temperature_to_state(air_temp_value)
    wind = wind_speed_to_state(wind_speed)
    pressure = pressure_to_state(pressure_value)

    return weather, weather_name, air_temp, air_temp_value, wind, wind_speed, pressure, pressure_value, season_name


# =====================================================
# 8) SULAMA MİKTARI FONKSİYONU
# =====================================================
# Q-Learning sadece sulama açık mı kapalı mı kararını verir.
# Verilecek su miktarı bu fonksiyonda belirlenir.


def water_amount_liter(soil, air_temp, wind, weather, action):
    """
    Sulama yapılırsa verilecek su miktarını litre olarak belirler.
    """

    # Sulama kapalıysa su verilmez.
    if action == 0:
        return 0

    # Yağmurlu veya karlı havada sulama yapılmaz.
    if weather in [0, 3]:
        return 0

    # Rüzgar yüksekse sulama verimsiz olacağı için su verilmez.
    if wind == 2:
        return 0

    # Toprak kuru ve hava sıcaksa fazla su verilir.
    if soil == 0 and air_temp == 2:
        return 70

    # Toprak kuruysa orta miktarda su verilir.
    if soil == 0:
        return 50

    # Toprak nemli ve hava sıcaksa az-orta su verilir.
    if soil == 1 and air_temp == 2:
        return 30

    # Toprak nemliyse az su verilir.
    if soil == 1:
        return 15

    # Toprak ıslaksa su verilmez.
    return 0


# =====================================================
# 9) YIL SONU DEĞERLENDİRME FONKSİYONU
# =====================================================

def evaluate_water_usage(total_water, tree_health):
    """
    Yıl sonunda su tüketimi ve ağaç sağlığına göre sistemin başarısını yorumlar.
    """

    print("\nSU KULLANIM DEĞERLENDİRMESİ")
    print("-" * 50)
    print("Yıllık toplam su kullanımı:", total_water, "Litre")
    print("Yıl sonu ağaç sağlığı:", round(tree_health, 1), "/ 100")

    if tree_health >= 85 and total_water <= 6000:
        print("Sonuç: Sistem başarılı. Ağaç sağlıklı tutulmuş ve su tasarrufu sağlanmıştır.")
    elif tree_health >= 85 and total_water > 6000:
        print("Sonuç: Ağaç sağlıklı ancak su kullanımı yüksek. Sistem daha tasarruflu hale getirilebilir.")
    elif tree_health < 85 and total_water <= 6000:
        print("Sonuç: Su tasarrufu sağlanmış ancak ağaç sağlığı düşmüştür. Sulama miktarı artırılmalıdır.")
    else:
        print("Sonuç: Hem su kullanımı yüksek hem de ağaç sağlığı düşük. Sistem iyileştirilmelidir.")


# =====================================================
# 10) 1 YILLIK SULAMA TAKİBİ
# =====================================================

# Başlangıç toprak nemi
soil_moisture_pct = 65.0

# Başlangıç toprak durumu
soil = soil_moisture_to_state(soil_moisture_pct)

# Başlangıç ağaç sağlığı
tree_health = 100.0

# Başlangıç depo su miktarı
remaining_water = 1500.0

# Toplam kullanılan su
total_water = 0.0

# Toplam sulama sayısı
irrigation_count = 0


# Grafiklerde kullanılacak listeler
days = []
water_list = []
health_list = []
soil_state_list = []
soil_pct_list = []
air_temp_list = []
wind_speed_list = []
pressure_value_list = []
action_list = []
remaining_water_list = []

# Tablo ve GIF için günlük kayıt listesi
year_log = []


for day in range(1, 366):

    # O güne ait sensör verileri üretilir.
    weather, weather_name, air_temp, air_temp_value, wind, wind_speed, pressure, pressure_value, season_name = generate_daily_sensor_data(day)

    # Yağmur ve kar olduğunda depo suyu artar.
    if weather_name == "Yağmurlu":
        remaining_water = min(3000, remaining_water + 200)
    elif weather_name == "Karlı":
        remaining_water = min(3000, remaining_water + 50)

    # Eğitilmiş Q-table'a göre en iyi aksiyon seçilir.
    action = np.argmax(Q[soil, weather, air_temp, wind, pressure])

    # Seçilen aksiyona göre su miktarı belirlenir.
    water = water_amount_liter(soil, air_temp, wind, weather, action)

    # Depoda yeterli su yoksa sadece depodaki kadar su verilebilir.
    water = min(water, remaining_water)

    # Depodan kullanılan su düşülür.
    remaining_water -= water

    # Toplam kullanılan su güncellenir.
    total_water += water

    # Sulama yapıldıysa sayaç artırılır.
    if water > 0:
        irrigation_count += 1

    # Ağaç sağlığı güncellenir.
    if soil == 0 and water == 0:
        tree_health -= 2
    elif soil == 2 and water > 0:
        tree_health -= 4
    elif soil == 0 and water > 0:
        tree_health += 0.3
    elif soil == 1 and water == 0:
        tree_health += 0.1

    # Depoda su tamamen bittiyse ağaç sağlığı olumsuz etkilenir.
    if remaining_water <= 0:
        tree_health -= 0.5

    # Ağaç sağlığı 0 ile 100 arasında tutulur.
    tree_health = max(0, min(100, tree_health))

    # Toprak nem yüzdesi güncellenir.
    if water > 0:
        soil_moisture_pct += water * 0.08
    else:
        evap = 1.0

        # Güneşli hava buharlaşmayı artırır.
        if weather_name == "Güneşli":
            evap += 2.0

        # Çok sıcak hava buharlaşmayı artırır.
        if air_temp_value > 30:
            evap += 1.5

        # Yüksek rüzgar buharlaşmayı artırır.
        if wind_speed > 25:
            evap += 1.0

        # Yağmur ve kar buharlaşma etkisini azaltır.
        if weather_name in ["Yağmurlu", "Karlı"]:
            evap -= 1.0

        soil_moisture_pct -= max(0.2, evap)

    # Yağmur ve kar toprak nemini artırır.
    if weather_name == "Yağmurlu":
        soil_moisture_pct += random.uniform(2.0, 5.0)
    elif weather_name == "Karlı":
        soil_moisture_pct += random.uniform(0.5, 2.0)

    # Toprak nemi 10 ile 100 arasında sınırlandırılır.
    soil_moisture_pct = max(10.0, min(100.0, soil_moisture_pct))

    # Yeni toprak nem yüzdesine göre toprak state güncellenir.
    soil = soil_moisture_to_state(soil_moisture_pct)

    # Günlük veriler kaydedilir.
    year_log.append([
        day,
        soil_states[soil],
        round(soil_moisture_pct, 1),
        weather_name,
        air_temp_value,
        wind_speed,
        pressure_value,
        wind_states[wind],
        pressure_states[pressure],
        actions[action],
        water,
        round(tree_health, 1),
        round(remaining_water, 1),
        season_name
    ])

    # Grafikler için veriler listelere eklenir.
    days.append(day)
    water_list.append(water)
    health_list.append(tree_health)
    soil_state_list.append(soil)
    soil_pct_list.append(soil_moisture_pct)
    air_temp_list.append(air_temp_value)
    wind_speed_list.append(wind_speed)
    pressure_value_list.append(pressure_value)
    action_list.append(action)
    remaining_water_list.append(remaining_water)


# =====================================================
# 11) METİNSEL ÇIKTI
# =====================================================

print("\n1 YILLIK SULAMA TAKİBİ TAMAMLANDI")
print("-" * 125)
print("Gün | Toprak | Nem % | Hava | Sıc. | Rüzgar | Basınç | Karar | Su(L) | Sağlık | Depo")
print("-" * 125)

# İlk 30 gün tablo halinde yazdırılır.
for row in year_log[:30]:
    print(
        f"{row[0]:3d} | {row[1]:6s} | %{row[2]:4.1f} | {row[3]:8s} | "
        f"{row[4]:3d}°C | {row[5]:2d} km/h | {row[6]:4d} hPa | "
        f"{row[9]:7s} | {row[10]:4.0f} | {row[11]:5.1f} | {row[12]:6.1f} L"
    )

print("-" * 125)
print("Yıllık toplam sulama sayısı:", irrigation_count)
print("Yıllık toplam su kullanımı:", round(total_water, 1), "Litre")
print("Yıl sonu zeytin ağacı sağlık değeri:", round(tree_health, 1), "/ 100")
print("Yıl sonu kalan su:", round(remaining_water, 1), "Litre")

# Yıl sonu sistem başarısı değerlendirilir.
evaluate_water_usage(total_water, tree_health)


# =====================================================
# 12) GRAFİKLER
# =====================================================

# Reward değerlerinin daha düzgün görünmesi için hareketli ortalama alınır.
window = 50

reward_smooth = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

# Q-Learning reward grafiği
plt.figure(figsize=(12, 5))
plt.plot(reward_smooth, linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Ortalama Reward")
plt.title("Q-Learning Reward Hareketli Ortalama")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "grafik_0_reward_smooth.png"), dpi=300, bbox_inches="tight")
plt.show()

# Günlük sulama miktarı grafiği
plt.figure(figsize=(12, 5))
plt.bar(days, water_list)
plt.xlim(1, 365)
plt.xticks([1, 50, 100, 150, 200, 250, 300, 365])
plt.xlabel("Gün")
plt.ylabel("Su Miktarı (Litre)")
plt.title("Günlük Sulama Miktarı")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "grafik_1_sulama_miktari.png"), dpi=300, bbox_inches='tight')
plt.show()

# Ağaç sağlığı grafiği
plt.figure(figsize=(12, 5))
plt.plot(days, health_list, linewidth=2)
plt.axhspan(85, 100, alpha=0.15)
plt.axhspan(50, 85, alpha=0.15)
plt.axhspan(0, 50, alpha=0.15)
plt.xlim(1, 365)
plt.xticks([1, 50, 100, 150, 200, 250, 300, 365])
plt.xlabel("Gün")
plt.ylabel("Ağaç Sağlığı")
plt.title("Zeytin Ağacı Sağlık Durumu")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "grafik_2_agac_sagligi.png"), dpi=300, bbox_inches='tight')
plt.show()

# Toprak nem yüzdesi grafiği
plt.figure(figsize=(12, 5))
plt.plot(days, soil_pct_list, linewidth=2)
plt.axhspan(0, 40, alpha=0.1)
plt.axhspan(40, 75, alpha=0.15)
plt.axhspan(75, 100, alpha=0.1)
plt.axhline(40, linestyle="--", label="Kuru sınırı")
plt.axhline(75, linestyle="--", label="Islak sınırı")
plt.xlim(1, 365)
plt.xticks([1, 50, 100, 150, 200, 250, 300, 365])
plt.xlabel("Gün")
plt.ylabel("Toprak Nem (%)")
plt.title("Toprak Nem Yüzdesi")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "grafik_3_toprak_nem_yuzdesi.png"), dpi=300, bbox_inches='tight')
plt.show()
# =====================================================
# 13) UZAKTAN İZLEME PANELİ - GIF ANİMASYONU
# =====================================================
#
# Bu bölümde sistemin günlük çalışma durumu görsel olarak gösterilir.
#
# Oluşturulan GIF içerisinde:
#
# - Hava durumu
# - Mevsim bilgisi
# - Toprak rengi
# - Ağaç sağlığı
# - Sulama durumu
# - Su deposu seviyesi
# - Rüzgar etkisi
# - Toplam su tüketimi
#
# gibi bilgiler animasyon halinde gösterilir.
# =====================================================

print("\nGIF oluşturuluyor...")

# Grafik penceresi oluşturulur.
fig, ax = plt.subplots(figsize=(14, 8))


# =====================================================
# FRAME GÜNCELLEME FONKSİYONU
# =====================================================
#
# update(frame) fonksiyonu her gün için ekranı yeniden çizer.
#
# frame değeri:
# 0 -> 1. gün
# 1 -> 2. gün
# ...
# 364 -> 365. gün
#
# Her frame’de:
# - hava durumu güncellenir
# - toprak görünümü değişir
# - sulama açık/kapalı görünür
# - depo seviyesi değişir
# =====================================================

def update(frame):

    # Önce eski çizimler temizlenir.
    ax.clear()

    # Günlük kayıt bilgileri alınır.
    row = year_log[frame]

    day = row[0]
    soil_state_str = row[1]
    moisture_pct = row[2]
    weather_name = row[3]
    air_temp_value = row[4]
    wind_speed = row[5]
    pressure_value = row[6]
    action_str = row[9]
    water_amount = row[10]
    health = row[11]
    rem_water = row[12]
    season_name = row[13]


    # =====================================================
    # ARKA PLAN RENGİ
    # =====================================================
    #
    # Hava durumuna göre ekran rengi değiştirilir.
    #
    # Güneşli -> Açık sarı
    # Bulutlu -> Açık gri
    # Karlı -> Soğuk gri
    # Yağmurlu -> Açık mavi
    # =====================================================

    if weather_name == "Güneşli":
        bg_color = "#fffde7"

    elif weather_name == "Bulutlu":
        bg_color = "#f5f5f5"

    elif weather_name == "Karlı":
        bg_color = "#eceff1"

    else:
        bg_color = "#e3f2fd"

    ax.set_facecolor(bg_color)


    # =====================================================
    # BAŞLIK VE GÜN BİLGİSİ
    # =====================================================

    ax.text(
        0.40,
        0.95,
        "UZAKTAN AKILLI SULAMA TAKİP SİSTEMİ",
        ha="center",
        fontsize=17,
        fontweight="bold",
        color="#33691e",
        zorder=10
    )

    ax.text(
        0.40,
        0.90,
        f"Gün: {day}/365 | Mevsim: {season_name} | Hava: {weather_name}",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#424242",
        zorder=10
    )


    # =====================================================
    # TOPRAK GÖRÜNÜMÜ
    # =====================================================
    #
    # Toprak durumuna göre renk değiştirilir.
    #
    # Kuru -> Açık kahverengi
    # Nemli -> Orta kahverengi
    # Islak -> Koyu kahverengi
    # =====================================================

    if soil_state_str == "Kuru":
        soil_color = "#bcaaa4"

    elif soil_state_str == "Nemli":
        soil_color = "#795548"

    else:
        soil_color = "#3e2723"

    # Toprak zemini çizilir.
    ax.add_patch(
        plt.Rectangle(
            (0.1, 0.1),
            0.6,
            0.3,
            color=soil_color
        )
    )


    # =====================================================
    # AĞAÇ SAĞLIK DURUMU
    # =====================================================
    #
    # Sağlığa göre yaprak rengi değiştirilir.
    #
    # Sağlıklı -> Yeşil
    # Orta -> Sarı-yeşil
    # Kötü -> Soluk renk
    # =====================================================

    if health > 80:
        leaf_color = "#558b2f"

    elif health > 50:
        leaf_color = "#9e9d24"

    else:
        leaf_color = "#827717"


    # =====================================================
    # ZEYTİN AĞAÇLARINI ÇİZME
    # =====================================================
    #
    # 3 adet ağaç çizilir.
    # Gövde çizgilerle,
    # yapraklar scatter ile oluşturulur.
    # =====================================================

    for tx in [0.25, 0.40, 0.55]:

        # Ağaç gövdesi
        ax.plot(
            [tx, tx],
            [0.4, 0.60],
            color="#5d4037",
            linewidth=15
        )

        # Yapraklar
        ax.scatter(
            [tx-0.05, tx+0.05, tx, tx-0.02, tx+0.02],
            [0.55, 0.55, 0.65, 0.60, 0.60],
            s=3000,
            color=leaf_color,
            alpha=0.9
        )

        # Yaz döneminde zeytinler gösterilir.
        if health > 60 and (150 <= day <= 300):

            ax.scatter(
                [tx-0.02, tx+0.02, tx+0.05, tx-0.05],
                [0.58, 0.62, 0.53, 0.56],
                s=80,
                color="#1b1b1b"
            )


    # =====================================================
    # SU DEPOSU ÇİZİMİ
    # =====================================================
    #
    # Depodaki kalan su miktarı görsel olarak gösterilir.
    # Su azaldıkça mavi alan küçülür.
    # =====================================================

    tank_x, tank_y, tank_w, tank_h = 0.82, 0.1, 0.12, 0.4

    # Depo çerçevesi
    ax.plot(
        [tank_x, tank_x, tank_x + tank_w, tank_x + tank_w],
        [tank_y + tank_h, tank_y, tank_y, tank_y + tank_h],
        color="#455a64",
        linewidth=4
    )

    # Mevcut su yüksekliği hesaplanır.
    current_tank_h = (rem_water / 3000.0) * tank_h

    # Su seviyesi çizilir.
    ax.add_patch(
        plt.Rectangle(
            (tank_x, tank_y),
            tank_w,
            current_tank_h,
            color="#29b6f6",
            alpha=0.8
        )
    )

    # Depo etiketi
    ax.text(
        tank_x + tank_w/2,
        tank_y - 0.04,
        "SU DEPOSU",
        ha="center",
        fontweight="bold",
        fontsize=12,
        color="#455a64",
        zorder=10
    )

    # Depodaki litre bilgisi
    ax.text(
        tank_x + tank_w/2,
        tank_y + max(current_tank_h/2, 0.05),
        f"{rem_water:.0f} L\n%{(rem_water/3000.0)*100:.0f}",
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
        fontsize=12,
        zorder=10
    )


    # =====================================================
    # BORULAR
    # =====================================================
    #
    # Su deposundan ağaçlara giden borular çizilir.
    # =====================================================

    ax.plot(
        [0.25, tank_x],
        [0.15, 0.15],
        color="#78909c",
        linewidth=5
    )

    for tx in [0.25, 0.40, 0.55]:

        ax.plot(
            [tx, tx],
            [0.15, 0.35],
            color="#78909c",
            linewidth=4
        )


    # =====================================================
    # SULAMA AKTİFSE SU AKIŞI GÖSTERİLİR
    # =====================================================

    if water_amount > 0:

        # Ana borudaki su akışı
        ax.plot(
            [0.25, tank_x],
            [0.15, 0.15],
            color="#29b6f6",
            linewidth=2,
            linestyle="--"
        )

        for tx in [0.25, 0.40, 0.55]:

            # Dikey borulardaki su akışı
            ax.plot(
                [tx, tx],
                [0.15, 0.35],
                color="#29b6f6",
                linewidth=2,
                linestyle="--"
            )

            # Su damlaları
            for i in range(3):

                ax.scatter(
                    [tx - 0.02 - i*0.015, tx + 0.02 + i*0.015],
                    [0.32 - i*0.03, 0.32 - i*0.03],
                    marker="v",
                    color="#0288d1",
                    s=100
                )

        action_text = f"SULAMA AÇIK\n({water_amount:.0f} L)"
        action_color = "#0288d1"

    else:

        action_text = "SULAMA KAPALI"
        action_color = "#e53935"


    # =====================================================
    # SULAMA DURUM YAZISI
    # =====================================================

    ax.text(
        0.72,
        0.25,
        action_text,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=action_color,
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor=action_color
        ),
        zorder=10
    )


    # =====================================================
    # HAVA DURUMU GÖRSELLERİ
    # =====================================================

    # Güneşli hava
    if weather_name == "Güneşli":

        ax.scatter(
            [0.40],
            [0.85],
            s=6000,
            color="#ffee58",
            zorder=2
        )

        ax.scatter(
            [0.40],
            [0.85],
            s=9000,
            color="#ffee58",
            alpha=0.3,
            zorder=1
        )


    # Bulutlu hava
    elif weather_name == "Bulutlu":

        for cx in [0.25, 0.40, 0.55]:

            ax.scatter(
                [cx-0.04, cx-0.01, cx+0.02, cx+0.05],
                [0.85-0.01, 0.85+0.02, 0.85+0.03, 0.85],
                s=[2000, 3500, 3000, 2000],
                color="#90a4ae",
                alpha=0.9
            )


    # Yağmurlu hava
    elif weather_name == "Yağmurlu":

        for cx in [0.25, 0.40, 0.55]:

            # Bulutlar
            ax.scatter(
                [cx-0.04, cx-0.01, cx+0.02, cx+0.05],
                [0.85-0.01, 0.85+0.02, 0.85+0.03, 0.85],
                s=[2000, 3500, 3000, 2000],
                color="#607d8b",
                alpha=0.9
            )

            # Yağmur çizgileri
            for i in range(4):

                ax.plot(
                    [cx - 0.02 + i*0.015, cx - 0.04 + i*0.015],
                    [0.80, 0.45],
                    color="#03a9f4",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7
                )


    # Karlı hava
    elif weather_name == "Karlı":

        for cx in [0.25, 0.40, 0.55]:

            # Bulutlar
            ax.scatter(
                [cx-0.04, cx-0.01, cx+0.02, cx+0.05],
                [0.85-0.01, 0.85+0.02, 0.85+0.03, 0.85],
                s=[2000, 3500, 3000, 2000],
                color="#cfd8dc",
                alpha=0.9
            )

            # Kar taneleri
            for i in range(3):

                ax.scatter(
                    [cx - 0.02 + i*0.02],
                    [0.65 - (i%2)*0.1],
                    marker="*",
                    color="#00bcd4",
                    s=150,
                    alpha=0.8
                )


    # =====================================================
    # RÜZGAR GÖSTERGESİ
    # =====================================================
    #
    # Rüzgar hızı yüksekse oklar çizilir.
    # =====================================================

    if wind_speed > 20:

        ax.arrow(
            0.10,
            0.78,
            0.18,
            0,
            width=0.005,
            head_width=0.03,
            head_length=0.03,
            color="#607d8b"
        )

        ax.arrow(
            0.12,
            0.73,
            0.15,
            0,
            width=0.004,
            head_width=0.025,
            head_length=0.025,
            color="#607d8b"
        )


    # =====================================================
    # BİTKİ DURUMU YORUMU
    # =====================================================

    plant_health_str = (
        "BİTKİ SAĞLIKLI"
        if health >= 85
        else "BİTKİ RİSKTE"
        if health >= 50
        else "BİTKİ SAĞLIKSIZ"
    )


    # =====================================================
    # ANLIK RAPOR PANELİ
    # =====================================================

    summary_text = (
        "ANLIK SİSTEM RAPORU\n"
        "------------------------\n"
        f"Hava Durumu  : {weather_name}\n"
        f"Hava Sıc.    : {air_temp_value}°C\n"
        f"Rüzgar       : {wind_speed} km/h\n"
        f"Basınç       : {pressure_value} hPa\n"
        f"Toprak Nemi  : %{moisture_pct:.1f}\n"
        f"Top. Su Tük. : {sum(water_list[:frame+1]):.0f} L\n"
        f"Ağaç Durumu  : %{health:.0f} ({plant_health_str})\n"
    )

    ax.text(
        0.74,
        0.95,
        summary_text,
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
        bbox=dict(
            facecolor="#f5f5f5",
            alpha=0.95,
            edgecolor="#bdbdbd",
            boxstyle="round,pad=0.5"
        ),
        zorder=10
    )


    # =====================================================
    # GRAFİK SINIRLARI
    # =====================================================

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Eksenler gizlenir.
    ax.axis("off")


# =====================================================
# ANİMASYON OLUŞTURMA
# =====================================================
#
# interval=500:
# Her frame 500 ms sürer.
#
# fps=2:
# GIF saniyede 2 frame hızında kaydedilir.
# =====================================================

ani = FuncAnimation(
    fig,
    update,
    frames=len(days),
    interval=500
)

# GIF dosyası kaydedilir.
gif_path = os.path.join(OUTPUT_DIR, "zeytin_agaci_sulama.gif")
ani.save(
    gif_path,
    writer=PillowWriter(fps=2)
)

print(f"GIF başarıyla kaydedildi: {gif_path}")