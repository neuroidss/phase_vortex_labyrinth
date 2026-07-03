# =====================================================================
# FILE: neuro_driver.py (ДРАЙВЕР СБОРА ДАННЫХ)
# =====================================================================
import asyncio
import threading
import time
import collections
import os
import numpy as np

try:
    from pylsl import StreamInlet, resolve_byprop
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

try:
    from bleak import BleakScanner, BleakClient
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

def get_ubuntu_hci_adapters():
    try:
        adapters = sorted(os.listdir('/sys/class/bluetooth/'))
        found = [a for a in adapters if a.startswith('hci')]
        if found:
            return found
    except Exception:
        pass
    return ['hci0']


class DecodedPacketQueue:
    """
    Высокопроизводительный буфер отложенного декодирования.
    Освобождает критический поток BLE-событий от ресурсоемкого разбора байт.
    """
    def __init__(self, slot_idx, hci_adapter, packet_counts_ref=None):
        self.raw_queue = collections.deque()
        self.last_seq = None
        self.lost_count = 0
        self.slot_idx = slot_idx
        self.hci_adapter = hci_adapter
        self.packet_counts_ref = packet_counts_ref

    def append(self, data):
        # Мгновенная вставка сырых байт в очередь без парсинга
        self.raw_queue.append(data)
        # Увеличиваем счетчик пакетов только для BLE, так как LSL инкрементирует его вручную
        if not isinstance(data, list) and self.packet_counts_ref is not None:
            self.packet_counts_ref[self.slot_idx] += 1

    def __len__(self):
        return len(self.raw_queue)

    def popleft(self):
        if not self.raw_queue:
            return None
            
        data = self.raw_queue.popleft()
        
        # Если данные уже декодированы (поток LSL), возвращаем как есть
        if isinstance(data, list):
            return data
            
        # Декодирование сырого пакета на лету в основном вычислительном потоке
        if len(data) != 51 or data[0] != 0xA0 or data[50] != 0xC0: 
            return [0.0] * 16
        
        seq_num = int(data[1])
        if self.last_seq is not None:
            diff = (seq_num - self.last_seq) % 256
            if diff != 1:
                lost = (diff - 1) % 256
                self.lost_count += lost
                # Логирование потерь выведено из прерывания во избежание блокировок GIL
                print(f"\n[!!! ПОТЕРЯ !!!] Слот {self.slot_idx} ({self.hci_adapter}) пропустил {lost} пакетов!")
                
        self.last_seq = seq_num
        
        channels = [0.0] * 16
        for i in range(8):
            val = (data[2 + i*3] << 16) | (data[2 + i*3 + 1] << 8) | data[2 + i*3 + 2]
            if val & 0x800000: 
                val -= 0x1000000
            channels[i] = float(val)
        for i in range(8):
            val = (data[26 + i*3] << 16) | (data[26 + i*3 + 1] << 8) | data[26 + i*3 + 2]
            if val & 0x800000: 
                val -= 0x1000000
            channels[i+8] = float(val)
            
        return channels


class BoardWorker(threading.Thread):
    def __init__(self, slot_idx, buffers, queues, last_seq_nums, lost_packet_counts, packet_counts, hci_adapter, driver=None):
        super().__init__(daemon=True)
        self.slot_idx = slot_idx
        self.buffers = buffers
        self.queues = queues
        self.last_seq_nums = last_seq_nums
        self.lost_packet_counts = lost_packet_counts
        self.packet_counts = packet_counts
        self.hci_adapter = hci_adapter 
        self.driver = driver
        
        self.mac_address = None
        self.is_connected = False
        self.lock = threading.Lock()
        
    def assign_device(self, mac):
        with self.lock:
            self.mac_address = mac

    def release_device(self):
        with self.lock:
            self.mac_address = None
            self.is_connected = False

    def run(self):
        core_id = (self.slot_idx + 1) % os.cpu_count()
        try:
            os.sched_setaffinity(0, {core_id})
            print(f"[HW-CORE] Слот {self.slot_idx} заблокирован на CPU Core {core_id} | Адаптер: {self.hci_adapter}")
        except Exception:
            pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.worker_loop())

    async def worker_loop(self):
        while True:
            # Опциональное отключение автоподключения BLE при активных LSL клиентах
            if self.driver and self.driver.disable_ble_on_lsl and len(self.driver.lsl_inlets) > 0:
                await asyncio.sleep(1.0)
                continue

            mac = None
            with self.lock:
                mac = self.mac_address

            if mac is None or self.is_connected:
                await asyncio.sleep(0.5)
                continue

            try:
                async with BleakClient(mac, timeout=10.0, adapter=self.hci_adapter) as client:
                    self.is_connected = True
                    print(f"[Слот {self.slot_idx} на {self.hci_adapter}] <<< ПОДКЛЮЧЕН: {mac} >>>")
                    await client.start_notify(DATA_CHAR_UUID, self.ble_callback)
                    
                    while client.is_connected and self.mac_address == mac:
                        # Принудительное отключение, если в процессе работы появился LSL и активна опция
                        if self.driver and self.driver.disable_ble_on_lsl and len(self.driver.lsl_inlets) > 0:
                            break
                        await asyncio.sleep(0.1)
                        
            except Exception:
                pass
                
            self.release_device()
            await asyncio.sleep(1.0)

    def ble_callback(self, sender, data):
        # Мгновенная регистрация данных в буфер без блокировки прерывания
        self.queues[self.slot_idx].append(data)


class RealNeuroDriver:
    def __init__(self, char_uuid=DATA_CHAR_UUID):
        self.char_uuid = char_uuid
        self.buffers = [np.zeros((16, 500), dtype=np.float32) for _ in range(5)]
        
        self.last_seq_nums = [None] * 5
        self.lost_packet_counts = [0] * 5
        self.packet_counts = [0] * 5  
        self.sps_metrics = [0.0] * 5  
        
        self.lsl_inlets = {}
        self.scanner_running = True
        
        # Опция отключения BLE автоподключений/сканирования при наличии активных потоков LSL
        self.disable_ble_on_lsl = True
        
        self.system_adapters = get_ubuntu_hci_adapters()
        print(f"[HW-INIT] Доступные адаптеры в Ubuntu: {self.system_adapters}")
        
        self.queues = []
        for i in range(5):
            assigned_adapter = self.system_adapters[i % len(self.system_adapters)]
            self.queues.append(DecodedPacketQueue(i, assigned_adapter, self.packet_counts))
        
        self.workers = []
        for i in range(5):
            assigned_adapter = self.system_adapters[i % len(self.system_adapters)]
            w = BoardWorker(i, self.buffers, self.queues, self.last_seq_nums, self.lost_packet_counts, self.packet_counts, assigned_adapter, driver=self)
            self.workers.append(w)
            w.start()
            
        self.sps_thread = threading.Thread(target=self._run_sps_calc, daemon=True)
        self.sps_thread.start()

    def get_slot_raw_data(self, slot_idx):
        q_len = len(self.queues[slot_idx])
        if q_len > 0:
            batch = [self.queues[slot_idx].popleft() for _ in range(q_len)]
            new_data = np.array(batch).T 
            
            if q_len >= 500:
                self.buffers[slot_idx] = new_data[:, -500:]
            else:
                self.buffers[slot_idx][:, :-q_len] = self.buffers[slot_idx][:, q_len:]
                self.buffers[slot_idx][:, -q_len:] = new_data
        return self.buffers[slot_idx]

    def get_slot_data(self, slot_idx, simulated_data):
        worker = self.workers[slot_idx]
        is_lsl_active = any(v == slot_idx for v in self.lsl_inlets.values())
        if not worker.is_connected and not is_lsl_active:
            return simulated_data 
        return self.get_slot_raw_data(slot_idx)

    def get_active_slots_data(self, fallback_sim):
        active_slots = []
        for i in range(5):
            is_ble_active = self.workers[i].is_connected
            is_lsl_active = any(v == i for v in self.lsl_inlets.values())
            if is_ble_active or is_lsl_active:
                active_slots.append(i)
            
        if not active_slots:
            return fallback_sim, [0], False
                
        data_list = []
        for slot_idx in active_slots:
            data_list.append(self.get_slot_data(slot_idx, None))
                
        return np.concatenate(data_list, axis=0), active_slots, True

    def _run_sps_calc(self):
        try: os.sched_setaffinity(0, {0})
        except: pass
        while self.scanner_running:
            time.sleep(1.0)
            for i in range(5):
                self.sps_metrics[i] = self.packet_counts[i]
                self.packet_counts[i] = 0

    def start_lsl_scanning_thread(self):
        if LSL_AVAILABLE:
            t = threading.Thread(target=self._lsl_scan_loop, daemon=True)
            t.start()

    def _lsl_scan_loop(self):
        try: os.sched_setaffinity(0, {os.cpu_count() - 1})
        except: pass
        
        print("[LSL] Поиск сетевых трансляций...")
        while self.scanner_running:
            try:
                streams = resolve_byprop('type', 'EEG', timeout=1.0)
                for stream in streams:
                    if stream.name() not in self.lsl_inlets:
                        assigned_slots = list(self.lsl_inlets.values()) + [w.slot_idx for w in self.workers if w.is_connected]
                        free_slots = [i for i in range(5) if i not in assigned_slots]
                        
                        if free_slots:
                            slot_idx = free_slots[0]
                            inlet = StreamInlet(stream)
                            self.lsl_inlets[stream.name()] = slot_idx
                            print(f"[LSL] ОБНАРУЖЕН ПОТОК. Подключен к Слоту {slot_idx}.")
                            t = threading.Thread(target=self._lsl_pull_loop, args=(inlet, slot_idx), daemon=True)
                            t.start()
            except Exception: pass
            time.sleep(3.0)

    def _lsl_pull_loop(self, inlet, slot_idx):
        while self.scanner_running:
            chunk, timestamps = inlet.pull_chunk(max_samples=25)
            if timestamps:
                for sample in chunk:
                    self.queues[slot_idx].append(sample)
                    self.packet_counts[slot_idx] += 1
            time.sleep(0.004)

    def start_ble_scanning_thread(self, loop=None):
        if BLE_AVAILABLE and self.char_uuid:
            t = threading.Thread(target=self._scanner_thread_loop, daemon=True)
            t.start()

    def _scanner_thread_loop(self):
        try: os.sched_setaffinity(0, {0})
        except: pass
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ble_scan_loop())

    async def _ble_scan_loop(self):
        print(f"[SCANNER] Мастер-сканер запущен...")
        async def detection_callback(device, advertisement_data):
            # Отменяем регистрацию новых устройств BLE, если LSL активен и включена опция
            if self.disable_ble_on_lsl and len(self.lsl_inlets) > 0:
                return

            mac = device.address
            uuids = advertisement_data.service_uuids if advertisement_data.service_uuids else []
            if any(SERVICE_UUID.lower() in u.lower() for u in uuids):
                assigned_macs = set(w.mac_address for w in self.workers if w.mac_address is not None)
                if mac not in assigned_macs:
                    for w in self.workers:
                        if w.mac_address is None:
                            w.assign_device(mac)
                            break

        while self.scanner_running:
            # Прерываем сканирование BLE, если LSL активен и включена опция
            if self.disable_ble_on_lsl and len(self.lsl_inlets) > 0:
                await asyncio.sleep(1.0)
                continue

            try:
                async with BleakScanner(detection_callback=detection_callback, service_uuids=[SERVICE_UUID], adapter=self.system_adapters[0]):
                    await asyncio.sleep(4.0)
            except Exception:
                await asyncio.sleep(1.0)
