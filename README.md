# Twin-B Model - Building Simulation Platform

แพลตฟอร์มจำลองอาคารแบบ agent-based โดยใช้ **Mesa** และ **EnergyPlus** เพื่อศึกษาพฤติกรรมของผู้พักอาศัยและการควบคุมสภาพอากาศภายในใช้ประโยชน์จากการคำนวณแบบเวกเตอร์ที่รวดเร็วของ **PyTorch** **torch.distributed** สำหรับการประเมินสถานการณ์แบบขนานและจัดการผลลัพธ์และพารามิเตอร์ผ่านไฟล์ CSV และ config.yaml เพื่อความสามารถในการปรับขยายและขยายได้

---

## สารบัญ

- [ฟีเจอร์](#features)
- [โครงสร้างโปรเจกต์](#project-structure)
- [ข้อกำหนด](#requirements)
- [การติดตั้ง](#installation)
- [การตั้งค่า](#configuration)
- [วิธีใช้](#usage)
- [รายละเอียดโครงสร้างโปรเจกต์](#project-structure-details)
- [การมีส่วนร่วม](#contributing)
- [ใบอนุญาต](#license)

---

## ฟีเจอร์

- **Agent-Based Modeling**: จำลองพฤติกรรมผู้พักอาศัยโดยใช้เฟรมเวิร์ก Mesa
- **EnergyPlus Integration**: จำลองพลังงานอาคารที่มีความแม่นยำสูง
- **Distributed Computing**: การประเมินสถานการณ์แบบขนานโดยใช้ PyTorch distributed
- **Flexible Configuration**: การจัดการการตั้งค่าที่ยืดหยุ่นบนพื้นฐาน YAML
- **Multi-Zone Support**: รองรับเค้าโครงอาคารที่ซับซ้อนพร้อมหลายโซน
- **Extensible Architecture**: เพิ่มพฤติกรรมและการควบคุม agent ใหม่ได้อย่างง่ายดาย

---

## Project Structure

```
Twin-B/
│
├── main.py                              # Main entry point for simulation
├── model.py                             # BuildingModel and agent definitions
├── agent.py                             # Agent behavior and logic
├── utils.py                             # Utility functions
├── config.yaml                          # Simulation configuration
├── agents.json                          # Agent definitions
├── agents_schedule.json                 # Agent scheduling
├── job.slurm                            # SLURM job submission script
├── README.md                            # This file
│
└── EnergyPlus_BP_Boonchoo/              # EnergyPlus simulation folder
    ├── Energy+.idd                      # EnergyPlus data dictionary
    ├── expanded.idf                     # Building energy model file
    └── in.epw                           # Weather data file
```

### คำอธิบายไฟล์

| File/Folder | วัตถุประสงค์ |
|---|---|
| **main.py** | จุดเข้า; เริ่มต้นการคำนวณแบบกระจาย โหลดการตั้งค่า จัดการเวิร์กโฟลว์จำลอง |
| **model.py** | กำหนดคลาส `BuildingModel` (Mesa model) และจัดการปฏิสัมพันธ์ agent/zone |
| **agent.py** | ใช้พฤติกรรม agent ตรรกะการตัดสินใจ และการเปลี่ยนแปลงสถานะ |
| **utils.py** | ฟังก์ชันช่วยสำหรับการประมวลผลข้อมูล การบันทึก และการดำเนินการ I/O |
| **config.yaml** | พารามิเตอร์จำลอง: จำนวนขั้นตอน โซน โซนสภาพอากาศ การตั้งค่าโมเดล |
| **agents.json** | คำจำกัดความ Agent: ประเภท พารามิเตอร์เริ่มต้น ความสามารถ |
| **agents_schedule.json** | ตารางเวลา/ตารางเวลาสำหรับกิจกรรม agent และรูปแบบการใช้งาน |
| **job.slurm** | สคริปต์ส่งงาน SLURM สำหรับการดำเนินการ HPC cluster |
| **EnergyPlus_BP_Boonchoo/** | ข้อมูลจำลอง EnergyPlus รวมถึงเรขาคณิตอาคาร (IDF) และข้อมูลสภาพอากาศ (EPW) |

---

## ข้อกำหนด

- **Python**: 3.8 ขึ้นไป
- **Operating System**: Linux (แนะนำ), Windows หรือ macOS
- **RAM**: อย่างน้อย 8GB (16GB+ แนะนำสำหรับการจำลองขนาดใหญ่)
- **GPU** (ตัวเลือก): GPU ที่รองรับ CUDA สำหรับการจำลองแบบกระจาย

### Python Dependencies

```
mesa>=1.2.0
PyYAML>=6.0
pandas>=1.3.0
torch>=2.0.0
pyenergyplus>=0.1.0
numpy>=1.21.0
```

---

## การติดตั้ง

### 1. Clone หรือ Download โปรเจกต์

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd Twin-B

# Or extract from compressed file
tar -xzf Twin-B.tar.gz
cd Twin-B
```

### 2. สร้าง Virtual Environment (แนะนำ)

**บน Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**บน Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. ติดตั้ง Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install required packages
pip install -r requirements.txt
```

หรือติดตั้งแยกตามรายการ:
```bash
pip install mesa PyYAML pandas torch numpy
```

### 4. Download และ Setup EnergyPlus

โปรเจกต์ต้องใช้ EnergyPlus 25.1.0 เส้นทางไบนารีจะอ้างอิงใน `main.py`:

```bash
# ดาวน์โหลด EnergyPlus สำหรับแพลตฟอร์มของคุณจาก:
# https://energyplus.net/downloads

# แตกไฟล์ไปยังไดเรกทอรีโปรเจกต์:
# EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64/
```

ตรวจสอบว่าเส้นทางไบนารีใน `main.py` ตรงกับระบบของคุณ:
```python
os.environ["ENERGYPLUS_EXE"] = "./EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64/energyplus"
```

### 5. ตรวจสอบการติดตั้ง

```bash
python main.py --help
```

---

## การตั้งค่า

### config.yaml

ไฟล์การตั้งค่าจำลองหลักพร้อมส่วนต่อไปนี้:

```yaml
mesa:
  steps: 96                    # Number of simulation steps
  zones:                        # Building zones to simulate
    - "Zone_Fire-escape_R"
    - "Zone_Restroom_7503"
    # ... more zones
```

**พารามิเตอร์หลัก:**
- **steps**: ขั้นตอนเวลาจำลอง (96 = 24 ชั่วโมงพร้อมช่วงเวลา 15 นาที)
- **zones**: รายการโซนอาคารจากไฟล์ IDF
- **climate_zone**: การจำแนกโซนสภาพอากาศของอาคาร (ตัวเลือก)

### agents.json & agents_schedule.json
- **agents.json**: กำหนดประเภท agent และคุณสมบัติของ agent
- **agents_schedule.json**: ระบุตารางเวลาการใช้งานและกิจกรรม

---

## วิธีใช้

### รันการจำลอง Single

```bash
python main.py
```

### รันด้วยการตั้งค่าแบบกำหนดเอง

```bash
python main.py --config custom_config.yaml
```

### รันการจำลองแบบกระจาย (Multi-GPU/Multi-Node)

```bash
# Single node, multi-GPU
python -m torch.distributed.launch --nproc_per_node=4 main.py

# Multi-node using SLURM
sbatch job.slurm
```

### ตรวจสอบผลลัพธ์

ไฟล์ผลลัพธ์มักจะบันทึกไปที่:
```
outputs/
├── simulation_results/     # สถานะ Agent และสภาพโซน
├── outEnergyPlusBoonchoo_<scenario>_<policy>/     # ข้อมูลพลังงานอาคาร
└── 
```

---

## รายละเอียดโครงสร้างโปรเจกต์

### เวิร์กโฟลว์จำลอง

```
1. โหลดการตั้งค่า (config.yaml)
2. เริ่มต้น BuildingModel (mesa)
3. เชื่อมต่อกับ EnergyPlus
4. สร้าง agents ตามไฟล์ agents.json
5. รันลูปจำลอง:
   - ขั้นตอน Agent (การตัดสินใจ)
   - อัปเดตสภาพแวดล้อม
   - ขั้นตอน EnergyPlus (จำลองพลังงาน)
   - รวบรวมข้อมูล
6. บันทึกผลลัพธ์เป็น CSV
```

### คลาสหลัก

- **BuildingModel** (model.py): Mesa Model จัดการ agents และสภาพแวดล้อม
- **Agent** (agent.py): Mesa Agent แสดงถึงผู้พักอาศัยพร้อมพฤติกรรม
- **ZoneController**: จัดการการควบคุม HVAC และการควบคุมระดับโซน
- **OccupancyAgent**: จำลองพฤติกรรมผู้พักอาศัยและความเป็นใจคว้ามความร้อน

---

## ตัวอย่างเวิร์กโฟลว์

```bash
# 1. ตั้งค่าสภาพแวดล้อม
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. ดาวน์โหลด EnergyPlus และแตกไฟล์

# 4. ตรวจสอบและปรับ config.yaml

# 5. รันจำลอง
python main.py

# 6. วิเคราะห์ผลลัพธ์ในโฟลเดอร์ outputs/
```

---

## การสนับสนุนและเอกสาร

- **Mesa Documentation**: https://mesa.readthedocs.io/
- **EnergyPlus Documentation**: https://energyplus.net/documentation
- **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html

---


## References

- Mesa Framework: https://github.com/projectmesa/mesa
- EnergyPlus: https://energyplus.net/
- PyTorch: https://pytorch.org/
