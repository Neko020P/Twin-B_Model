# Twin-B Model - คู่มือการติดตั้งระบบ (Installation Guide)

บทความนี้เป็นคู่มืออย่างละเอียดสำหรับการติดตั้งและตั้งค่าระบบจำลองอาคาร Twin-B ให้พร้อมใช้งาน

---

## สารบัญ

1. [ข้อกำหนดของระบบ](#ข้อกำหนดของระบบ)
2. [สภาพแวดล้อม Python](#สภาพแวดล้อม-python)
3. [การติดตั้ง EnergyPlus](#การติดตั้ง-energyplus)
4. [การติดตั้งแพคเกจ Python](#การติดตั้งแพคเกจ-python)
5. [การตรวจสอบการติดตั้ง](#การตรวจสอบการติดตั้ง)
6. [การตั้งค่าเบื้องต้น](#การตั้งค่าเบื้องต้น)
7. [การเรียกใช้ครั้งแรก](#การเรียกใช้ครั้งแรก)
8. [การแก้ไขปัญหา](#การแก้ไขปัญหา)

---

## ข้อกำหนดของระบบ

### ความต้องการของฮาร์ดแวร์

| คุณสมบัติ | ข้อกำหนดขั้นต่ำ | แนะนำ |
|---------|---------|--------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 5 GB | 20 GB+ |
| **GPU** | ไม่จำเป็น | CUDA compatible (NVIDIA) |

### ความต้องการของซอฟต์แวร์

| ซอฟต์แวร์ | เวอร์ชั่น | หมายเหตุ |
|---------|---------|--------|
| **Python** | 3.8+ | แนะนำ 3.9, 3.10, 3.11 |
| **EnergyPlus** | 25.1.0 | ข้อบังคับ |
| **Git** | ใหม่ล่าสุด | สำหรับการดาวน์โหลด (ตัวเลือก) |

### ระบบปฏิบัติการที่รองรับ

- **Linux** (แนะนำ: Ubuntu 20.04 LTS, CentOS 7.9+)
- **Windows** (10, 11)
- **macOS** (Intel/Apple Silicon)

---

## สภาพแวดล้อม Python

### ขั้นตอนที่ 1: ตรวจสอบการติดตั้ง Python

**บน Linux/macOS:**
```bash
python3 --version
which python3
```

**บน Windows (PowerShell):**
```powershell
python --version
where.exe python
```

ต้องมี Python 3.8 ขึ้นไป

### ขั้นตอนที่ 2: สร้าง Virtual Environment

**บน Linux/macOS:**
```bash
# นำทางไปยังไดเรกทอรีโปรเจกต์
cd /path/to/Twin-B

# สร้าง virtual environment
python3 -m venv venv

# เปิดใช้งาน
source venv/bin/activate
```

**บน Windows (PowerShell):**
```powershell
# นำทางไปยังไดเรกทอรีโปรเจกต์
cd C:\path\to\Twin-B

# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน
.\venv\Scripts\Activate.ps1
```

> **หมายเหตุ**: หลังจากเปิดใช้งาน Virtual Environment จะปรากฏ `(venv)` ที่จุดเริ่มต้นของบรรทัดคำสั่ง

### ขั้นตอนที่ 3: ปรับปรุง pip

```bash
pip install --upgrade pip setuptools wheel
```

---

## การติดตั้ง EnergyPlus

### ขั้นตอนที่ 1: ดาวน์โหลด EnergyPlus 25.1.0

1. ไปที่เว็บไซต์ https://energyplus.net/downloads
2. เลือกเวอร์ชั่น **25.1.0**
3. เลือกระบบปฏิบัติการของคุณ:
   - **Linux**: `EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64.tar.gz`
   - **Windows**: `EnergyPlus-25.1.0-1c11a3d85f-Windows-x86_64.zip`
   - **macOS**: ดาวน์โหลดเวอร์ชั่นที่เหมาะสม

### ขั้นตอนที่ 2: แตกไฟล์

**บน Linux/macOS:**
```bash
# นำทางไปยังไดเรกทอรีโปรเจกต์
cd /path/to/Twin-B/src

# แตกไฟล์
tar -xzf EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64.tar.gz
```

**บน Windows (PowerShell):**
```powershell
# นำทางไปยังไดเรกทอรีโปรเจกต์
cd C:\path\to\Twin-B\src

# แตกไฟล์ (ใช้ 7-Zip หรือ WinRAR)
# หรือใช้ PowerShell
Expand-Archive -Path EnergyPlus-25.1.0-1c11a3d85f-Windows-x86_64.zip -DestinationPath .
```

### ขั้นตอนที่ 3: อัปเดตเส้นทางใน main.py

เปิดไฟล์ `main.py` และตรวจสอบ/อัปเดตเส้นทางไฟล์ EnergyPlus:

**สำหรับ Linux:**
```python
os.environ["ENERGYPLUS_EXE"] = "./EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64/energyplus"
```

**สำหรับ Windows:**
```python
os.environ["ENERGYPLUS_EXE"] = ".\\EnergyPlus-25.1.0-1c11a3d85f-Windows-x86_64\\energyplus.exe"
```

**สำหรับ macOS:**
```python
os.environ["ENERGYPLUS_EXE"] = "./EnergyPlus-25.1.0-1c11a3d85f-Darwin-x86_64/energyplus"
```

### ขั้นตอนที่ 4: ตรวจสอบการติดตั้ง EnergyPlus

**บน Linux/macOS:**
```bash
cd /path/to/Twin-B/src
./EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64/energyplus --version
```

**บน Windows:**
```powershell
cd C:\path\to\Twin-B\src
.\EnergyPlus-25.1.0-1c11a3d85f-Windows-x86_64\energyplus.exe --version
```

ควรแสดงเลขเวอร์ชั่น: `EnergyPlus, Version 25.1.0`

---

## การติดตั้งแพคเกจ Python

### ขั้นตอนที่ 1: สร้างไฟล์ requirements.txt (ถ้าไม่มี)

หากไม่มีไฟล์ `requirements.txt` ให้สร้างในไดเรกทอรี Twin-B/src ด้วยเนื้อหาดังนี้:

```txt
mesa>=1.2.0
PyYAML>=6.0
pandas>=1.3.0
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### ขั้นตอนที่ 2: ติดตั้งแพคเกจ

ตรวจสอบว่า Virtual Environment เปิดใช้งานแล้ว จากนั้น:

```bash
# ติดตั้งจากไฟล์ requirements.txt
pip install -r requirements.txt

# หรือติดตั้งแยกตามรายการ
pip install mesa PyYAML pandas torch numpy matplotlib
```

### ขั้นตอนที่ 3: ติดตั้งแพคเกจ PyEnergyPlus (ตัวเลือก)

```bash
pip install pyenergyplus
```

> **หมายเหตุ**: หากติดตั้ง PyTorch โดยมี GPU support เลือกคำสั่งเหมาะสม:
>
> **สำหรับ CUDA 12.1:**
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
>
> **สำหรับ CPU เท่านั้น:**
> ```bash
> pip install torch torchvision torchaudio
> ```

---

## การตรวจสอบการติดตั้ง

### ตรวจสอบแพคเกจ Python

```bash
# ตรวจสอบเวอร์ชั่นแพคเกจ
pip list | grep -E "mesa|PyYAML|pandas|torch|numpy"
```

### สคริปต์ตรวจสอบการติดตั้ง

สร้างไฟล์ `verify_installation.py` ในไดเรกทอรี Twin-B/src:

```python
#!/usr/bin/env python3

import sys
print("=" * 60)
print("Twin-B Installation Verification Script")
print("=" * 60)

packages = [
    ("mesa", "Mesa Agent-Based Modeling"),
    ("yaml", "PyYAML"),
    ("pandas", "Pandas"),
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
]

failed = []
for package, name in packages:
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name:.<40} version {version}")
    except ImportError:
        print(f"✗ {name:.<40} NOT INSTALLED")
        failed.append(name)

print("=" * 60)

# ตรวจสอบ EnergyPlus
import os
print("\nEnergyPlus Path Check:")
ep_path = os.environ.get("ENERGYPLUS_EXE", "NOT SET")
print(f"  ENERGYPLUS_EXE = {ep_path}")

if os.path.exists(ep_path):
    print(f"  ✓ EnergyPlus executable found")
else:
    print(f"  ✗ EnergyPlus executable NOT found")
    failed.append("EnergyPlus")

print("\n" + "=" * 60)
if failed:
    print(f"✗ Installation incomplete. Missing: {', '.join(failed)}")
    sys.exit(1)
else:
    print("✓ All dependencies installed successfully!")
    sys.exit(0)
```

เรียกใช้สคริปต์ตรวจสอบ:

```bash
python verify_installation.py
```

---

## การตั้งค่าเบื้องต้น

### ตรวจสอบไฟล์ตั้งค่า

ตรวจสอบความมีตัวอักษรต่อไปนี้อยู่ในไดเรกทอรี `Twin-B/src`:

- **config.yaml** - การตั้งค่าการจำลอง (จำนวนขั้นตอน โซน ฯลฯ)
- **agents.json** - คำนิยาม Agent
- **agents_schedule.json** - ตารางเวลา Agent
- **EnergyPlus_BP_Boonchoo/** - ข้อมูล EnergyPlus
  - `Energy+.idd`
  - `expanded.idf`
  - `in.epw`

### การแก้ไขการตั้งค่า

#### 1. config.yaml

สำหรับการเรียกใช้งานครั้งแรก ให้ปรับพารามิเตอร์บางอย่างเพื่อการทดสอบด่วน:

```yaml
mesa:
  steps: 960                    # จำนวนขั้นตอนจำลอง (ลดลงสำหรับการทดสอบ)
  zones: [...]                  # รายการโซน (อย่าแก้ไข)
  
simulation:
  distributed: false            # ปิด distributed computing สำหรับการทดสอบ
  num_workers: 1                # จำนวน worker processes
  batch_size: 1                 # ขนาด batch สำหรับการจำลอง
```

#### 2. agents.json

ตัวอย่าง agent definition:

```json
{
  "agents": [
    {
      "type": "occupant",
      "count": 5,
      "properties": {
        "preferred_temp": 25.0,
        "comfort_tolerance": 1.0,
        "heat_gain_watts": 100.0
      }
    }
  ]
}
```

#### 3. agents_schedule.json

ตัวอย่าง scheduling:

```json
{
  "schedule": {
    "weekday": {
      "07:00": "arrive",
      "17:00": "leave"
    },
    "weekend": {
      "09:00": "arrive",
      "18:00": "leave"
    }
  }
}
```

---

## การเรียกใช้ครั้งแรก

### ขั้นตอนที่ 1: เตรียม Environment

```bash
# เปิดใช้งาน virtual environment
# บน Linux/macOS:
source venv/bin/activate

# บน Windows:
.\venv\Scripts\Activate.ps1

# ตรวจสอบการติดตั้ง
python verify_installation.py
```

### ขั้นตอนที่ 2: การเรียกใช้งานทดสอบ

สำหรับการทดสอบครั้งแรก ให้ลดจำนวนขั้นตอนใน `config.yaml`:

```yaml
mesa:
  steps: 24  # แค่ 24 ขั้นตอน (1 วัน)
```

จากนั้นเรียกใช้:

```bash
python main.py
```

### ขั้นตอนที่ 3: ตรวจสอบผลลัพธ์

หากการเรียกใช้สำเร็จ คุณจะเห็น:

- ข้อมูลเกี่ยวกับการเชื่อมต่อ Distributed (หากเปิดใช้งาน)
- ความคืบหน้าจำลอง
- ผลลัพธ์ที่บันทึกไว้ (โดยปกติใน `outputs/` หรือ `results/`)

---

## การแก้ไขปัญหา

### ปัญหา: ModuleNotFoundError

**ข้อความ:** `ModuleNotFoundError: No module named 'mesa'`

**วิธีแก้ไข:**
1. ตรวจสอบว่า Virtual Environment เปิดใช้งาน: `which python` หรือ `where python` ควรแสดง `venv/bin/python`
2. ติดตั้งแพคเกจใหม่: `pip install mesa`

---

### ปัญหา: EnergyPlus executable not found

**ข้อความ:** `FileNotFoundError: EnergyPlus executable not found`

**วิธีแก้ไข:**
1. ตรวจสอบว่า EnergyPlus แตกไฟล์ถูกต้อง:
   ```bash
   ls -la EnergyPlus-25.1.0-*  # Linux/macOS
   dir EnergyPlus-25.1.0-*     # Windows
   ```
2. อัปเดตเส้นทาง `ENERGYPLUS_EXE` ใน `main.py` ให้ตรงกับเส้นทางจริง
3. ตรวจสอบสิทธิ์การเข้าถึง (Linux/macOS):
   ```bash
   chmod +x ./EnergyPlus-25.1.0-*/energyplus
   ```

---

### ปัญหา: CUDA/GPU errors

**ข้อความ:** `RuntimeError: CUDA out of memory` หรือเกี่ยวกับ GPU

**วิธีแก้ไข:**
1. ลดขนาด batch ใน `config.yaml`:
   ```yaml
   batch_size: 1
   ```
2. ปิด GPU ใช้ CPU แทน:
   ```python
   # ใน main.py
   torch.cuda.is_available = lambda: False
   ```
3. ตรวจสอบ GPU memory:
   ```bash
   nvidia-smi  # Linux/macOS
   ```

---

### ปัญหา: ข้อผิดพลาดการตั้งค่า Distributed

**ข้อความ:** `RuntimeError: Distributed initialization failed`

**วิธีแก้ไข:**
1. สำหรับการทดสอบครั้งแรก ให้ปิด distributed computing ใน `config.yaml`:
   ```yaml
   distributed: false
   ```
2. ตรวจสอบตัวแปร environment:
   ```bash
   echo $RANK $WORLD_SIZE $LOCAL_RANK
   ```

---

### ปัญหา: YAML configuration errors

**ข้อความ:** `yaml.YAMLError` หรือ `ValueError: Invalid configuration`

**วิธีแก้ไข:**
1. ตรวจสอบไวยากรณ์ YAML ใน `config.yaml`:
   - ใช้เว็บไซต์เช่น https://www.yamllint.com/ เพื่อตรวจสอบ
   - ตรวจสอบความถูกต้องของ indentation (ใช้ 2 หรือ 4 spaces, ไม่ใช้ tabs)
2. ตรวจสอบรูปแบบ JSON ใน `agents.json` และ `agents_schedule.json`

---

## ถัดไป

หลังจากติดตั้งสำเร็จ โปรดอ้างอิง:

- **README.md** สำหรับข้อมูลทั่วไปและข้อมูลเพิ่มเติม
- **Source Code** สำหรับรายละเอียดการใช้งาน API
- **Configuration Files** สำหรับการตั้งค่า simulation

---

## การติดต่อและการสนับสนุน

หากมีปัญหาหรือข้อสงสัย:

1. ตรวจสอบไฟล์ log output
2. รวบรวมข้อมูล: OS, Python version, error messages
3. ติดต่อทีมพัฒนา

---

**เวอร์ชั่น:** 1.0  
**วันที่อัปเดต:** December 2025  
**ความเข้ากันได้:** Python 3.8+, EnergyPlus 25.1.0
