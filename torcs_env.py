import numpy as np
import gymnasium as gym
from gymnasium import spaces
from snakeoil import Client

class TorcsEnv(gym.Env):
    """
    Wrapper Gym dla TORCS oparty na kliencie snakeoil.
    
    Przestrzeń obserwacji: 29 sensorów (prędkości, pozycja, kąt, track edges)
    Przestrzeń akcji:      3 wartości ciągłe (gaz, hamulec, skręt)
    """

    # Liczba sensorów odległości od krawędzi toru
    TRACK_SENSORS = 19

    def __init__(self, port=3001, vision=False):
        super().__init__()
        self.port = port
        self.vision = vision
        self.client = None        # Połączenie z TORCS — tworzymy w reset()
        self.terminal_judge_start = 100   # Od którego kroku sprawdzamy czy auto utknęło
        self.time_step = 0

        # === PRZESTRZEŃ AKCJI ===
        # Agent steruje trzema wartościami, każda w zakresie [-1, 1]
        # Indeks 0: steer  — skręt     (-1 = pełny lewo, +1 = pełny prawo)
        # Indeks 1: accel  — gaz       (-1 = brak, +1 = pełny gaz)*
        # Indeks 2: brake  — hamulec   (-1 = brak, +1 = pełny hamulec)*
        # (* przeskalujemy do [0,1] przy wysyłaniu do TORCS)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # === PRZESTRZEŃ OBSERWACJI ===
        # Wszystkie wartości znormalizowane do [-1, 1] lub [0, 1]
        # Szczegóły przy metodzie _get_obs()
        obs_dim = 30  # wyjaśnimy dokładnie poniżej
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Zaczyna nowy epizod.
        Tworzy (lub restartuje) połączenie z TORCS i zwraca pierwszy stan.
        """
        super().reset(seed=seed)

        if self.client is None or self.client.so is None:
            # Pierwsze uruchomienie — stwórz klienta (on sam uruchomi TORCS)
            self.client = Client(p=self.port, vision=self.vision)
        else:
            # Kolejny epizod — wyślij meta=1 żeby TORCS zresetował wyścig
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()
            self.client.R.d['meta'] = 0

        self.time_step = 0
        self._stuck_count = 0
        self._prev_damage = 0

        # Pobierz pierwszy stan z serwera
        self.client.get_servers_input()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.time_step += 1

        # 1. Prześlij akcję do TORCS
        self._apply_action(action)
        self.client.respond_to_server()

        # 2. Pobierz nowy stan
        self.client.get_servers_input()

        # Sprawdź czy klient nie rozłączył się (***restart*** lub ***shutdown***)
        if self.client.so is None:
            # TORCS rozłączył się — zakończ epizod
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -1.0, True, False, {}

        S = self.client.S.d

        # 3. Oblicz nagrodę
        reward = self._compute_reward(S)

        # 4. Sprawdź czy epizod się skończył
        terminated = self._is_terminal(S)
        truncated = False

        obs = self._get_obs()
        info = {
            'speed': S.get('speedX', 0),
            'trackPos': S.get('trackPos', 0),
            'angle': S.get('angle', 0),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.client:
            self.client.shutdown()
            self.client = None





    # =========================================================
    # Metody pomocnicze — wypełnimy je w następnych krokach
    # =========================================================





    #==========================================================
    # GET OBSERVATION
    #==========================================================

    def _get_obs(self):
        """
        Zamienia surowe dane z TORCS na znormalizowany wektor obserwacji.
        
        Każda wartość jest przeskalowana do [-1, 1] lub [0, 1] tak żeby
        sieć neuronowa dostawała wartości w podobnej skali.
        """
        S = self.client.S.d

        # Prędkości — normalizujemy przez maksymalne sensowne wartości
        # speedX: do przodu, max ~300 km/h w TORCS
        # speedY: boczna, max ~50 km/h (poślizg)
        # speedZ: pionowa, max ~50 km/h (nierówności)
        speed_x = np.clip(S.get('speedX', 0) / 300.0, -1, 1)
        speed_y = np.clip(S.get('speedY', 0) / 50.0,  -1, 1)
        speed_z = np.clip(S.get('speedZ', 0) / 50.0,  -1, 1)

        # Pozycja na torze — już w [-1, 1], ale clipujemy na wypadek
        # wartości spoza toru (agent wylatuje — może być >1 lub <-1)
        track_pos = np.clip(S.get('trackPos', 0), -1, 1)

        # Kąt między autem a osią toru — w radianach, zakres [-π, π]
        # Dzielimy przez π żeby dostać [-1, 1]
        angle = np.clip(S.get('angle', 0) / np.pi, -1, 1)

        # 19 sensorów odległości od krawędzi — wartości w metrach (0 do ~200m)
        # Normalizujemy przez 200, clipujemy do [0, 1]
        track_sensors = S.get('track', [0] * 19)
        if len(track_sensors) < 19:
            track_sensors = [0] * 19   # zabezpieczenie na wypadek błędu
        track_norm = np.clip(
            np.array(track_sensors, dtype=np.float32) / 200.0,
            0, 1
        )

        # Obroty kół — normalizujemy przez 100 rad/s (max przy pełnym gazie)
        # Różnice między kołami mówią agentowi o poślizgu
        wheel_spin = S.get('wheelSpinVel', [0, 0, 0, 0])
        if len(wheel_spin) < 4:
            wheel_spin = [0, 0, 0, 0]
        wheel_norm = np.clip(
            np.array(wheel_spin, dtype=np.float32) / 100.0,
            -1, 1
        )

        # RPM silnika — max ~10000 obr/min w TORCS
        rpm = np.clip(S.get('rpm', 0) / 10000.0, 0, 1)

        # Bieg — od -1 (wsteczny) do 6, normalizujemy do [-1, 1]
        gear = np.clip(S.get('gear', 0) / 6.0, -1, 1)

        # Składamy wszystko w jeden wektor
        obs = np.concatenate([
            [speed_x, speed_y, speed_z],   # 3
            [track_pos],                    # 1
            [angle],                        # 1
            track_norm,                     # 19
            wheel_norm,                     # 4
            [rpm],                          # 1
            [gear],                         # 1
        ]).astype(np.float32)               # łącznie: 30

        return obs

    
    #==========================================================
    # APPLY ACTION
    #==========================================================


    def _apply_action(self, action):
        S = self.client.S.d
        R = self.client.R.d

        accel = np.clip((action[1] + 1) / 2.0, 0, 1)
        brake = np.clip((action[2] + 1) / 2.0, 0, 1)

        # net = accel - brake

        # if net > 0:
        #     R['accel'] = max(net, 0.2)  # minimum 0.2 żeby auto się ruszało
        # else:
        #     R['accel'] = 0.0

        R['steer'] = np.clip(action[0], -1, 1)
        R['gear'] = self._auto_gear(S)
        R['accel'] = accel
        R['brake'] = brake
        R['clutch'] = 0
        R['meta'] = 0



    def _auto_gear(self, S):
        """
        Automatyczna skrzynia biegów oparta na prędkości.
        Prosta heurystyka — agent nie traci zasobów na uczenie biegów.
        """
        speed = S.get('speedX', 0)
        gear = int(S.get('gear', 1))

        # Progi zmiany biegów w górę
        up_thresholds   = [60, 100, 140, 180, 220]
        # Progi zmiany biegów w dół (trochę niższe żeby uniknąć oscylacji)
        down_thresholds = [40,  80, 120, 160, 200]

        if gear < 6 and speed > up_thresholds[gear - 1]:
            return gear + 1
        elif gear > 1 and speed < down_thresholds[gear - 2]:
            return gear - 1

        return max(1, gear)  # nigdy nie wróć do 0 (neutralny)



    #==========================================================
    # COMPUTE REWARD
    #==========================================================

    def _compute_reward(self, S):
        """
        Funkcja nagrody — serce całego systemu RL.
        
        Składniki:
            + prędkość wzdłuż osi toru  (chcemy jechać szybko i prosto)
            - kara za zjazd z toru      (chcemy być blisko środka)
            - kara za kąt               (chcemy jechać prosto, nie bokiem)
            - kara za kolizję           (nie uderzać w bariery)
        """
        speed   = S.get('speedX', 0)
        angle   = S.get('angle', 0)
        track_pos = S.get('trackPos', 0)
        damage  = S.get('damage', 0)

        # --- Składnik 1: prędkość wzdłuż osi toru ---
        # cos(angle) ≈ 1 gdy jedzie prosto, maleje gdy jedzie bokiem
        # Dzielimy przez 300 żeby znormalizować do ~[0, 1]
        reward_speed = speed * np.cos(angle) / 300.0

        # --- Składnik 2: kara za zjazd z toru ---
        # trackPos=0 → kara=0, trackPos=±1 → kara=1
        # Kwadrat = kara nieliniowa, mocno rośnie przy krawędzi
        penalty_pos = track_pos ** 2

        # --- Składnik 3: kara za kąt ---
        # Normalizujemy przez π, podnosimy do kwadratu
        # agent jadący bokiem (angle=π/2) dostaje karę ~0.25
        penalty_angle = (angle / np.pi) ** 2

        # --- Składnik 4: kara za kolizję ---
        # damage rośnie gdy auto uderza w bariery
        # Śledzimy zmianę damage między krokami
        prev_damage = getattr(self, '_prev_damage', 0)
        damage_delta = max(0, damage - prev_damage)
        self._prev_damage = damage

        penalty_damage = damage_delta * 0.1

        #### NAGRODA ####

        reward = (
            1.0 * reward_speed
            - 0.5 * penalty_pos
            - 0.2 * penalty_angle
            - 1.0 * penalty_damage
        )

        return float(reward)


    #==========================================================
    # IS TERMINAL
    #==========================================================

    def _is_terminal(self, S):
        """
        Sprawdza czy epizod powinien się zakończyć.
        
        Trzy warunki:
            1. Wyjazd z toru      (trackPos poza [-1, 1])
            2. Auto utknęło       (niska prędkość przez wiele kroków)
            3. Poważna kolizja    (damage przekroczył próg)
        """
        speed     = S.get('speedX', 0)
        track_pos = S.get('trackPos', 0)
        damage    = S.get('damage', 0)

        # --- Warunek 1: wyjazd z toru ---
        # Małe przesunięcie ponad 1.0 żeby dać agentowi chwilę
        # na korektę zanim przerwiemy epizod
        if abs(track_pos) > 1.1:
            print(f"[TERMINAL] Wyjazd z toru: trackPos={track_pos:.2f}")
            return True

        # --- Warunek 2: auto utknęło ---
        # Sprawdzamy dopiero po terminal_judge_start krokach
        # żeby dać agentowi czas na rozpędzenie się na starcie
        if self.time_step > self.terminal_judge_start:
            if speed < 1.0:   # poniżej 5 km/h = praktycznie stoi
                self._stuck_count = getattr(self, '_stuck_count', 0) + 1
            else:
                self._stuck_count = 0

            # Kończymy dopiero gdy stoi przez 30 kolejnych kroków
            # (~0.6 sekundy) — jeden zły krok to jeszcze nie problem
            if self._stuck_count > 30:
                print(f"[TERMINAL] Auto utknęło: speed={speed:.1f} przez {self._stuck_count} kroków")
                self._stuck_count = 0
                return True

        # --- Warunek 3: poważna kolizja ---
        if damage > 5000:
            print(f"[TERMINAL] Kolizja: damage={damage:.0f}")
            return True

        return False