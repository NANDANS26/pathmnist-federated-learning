# PHASE 0: COMMANDS AND VERIFICATION AUDIT LOG

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Dept. of CSE, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Host Shell:** Windows PowerShell / Antigravity Terminal  
**Working Directory:** `d:\NANDAN.S\ENGINEERING\GITHUB ISSUES`  
**Date:** November 6, 2026  

---

## 1. Overview of Executed Operations

During Phase 0, all system commands were executed strictly for read-only environment inspection, repository acquisition, and code auditing. No unauthorized installations, external downloads (other than the official git clone), or code modifications were performed.

---

## 2. Command Execution Log

### Command 1: Workspace & Current Directory Check
- **Command:** `Get-Location; Get-ChildItem`
- **Purpose:** Verify initial working directory status.
- **Exit Code:** `0` (Success)
- **Result:** Path verified as `D:\NANDAN.S\ENGINEERING\GITHUB ISSUES`. Directory was initially clean.

### Command 2: Environment Diagnostics (Initial Composite Script)
- **Command:** `powershell -Command "Write-Host '=== OS VERSION ==='; Get-CimInstance ..."`
- **Purpose:** Collect composite hardware and runtime diagnostics.
- **Exit Code:** `1` (Parser Error)
- **Issue Encountered:** PowerShell `$_.Property` token inside double-quoted string was stripped by the sub-shell argument parser, causing a syntax exception (`Missing ')' in method call`).
- **Remediation:** Switched immediately to direct, unnested PowerShell CIM queries.

### Command 3: Operating System & Physical RAM Telemetry
- **Command:** `Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory`
- **Purpose:** Retrieve OS build, 64-bit architecture, total physical RAM, and free memory.
- **Exit Code:** `0` (Success)
- **Verified Output:**
  - OS: `Microsoft Windows 10 Pro (Version 10.0.19045, 64-bit)`
  - TotalVisibleMemorySize: `4,114,004 KB` (~3.92 GB / 4.0 GB physical RAM)
  - FreePhysicalMemory: `593,924 KB` (~580 MB free RAM)

### Command 4: Processor Hardware Telemetry
- **Command:** `Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed`
- **Purpose:** Detect CPU model, core topology, and frequency.
- **Exit Code:** `0` (Success)
- **Verified Output:**
  - Name: `Intel(R) Pentium(R) CPU G3240 @ 3.10GHz`
  - NumberOfCores: `2`
  - NumberOfLogicalProcessors: `2`
  - MaxClockSpeed: `3100 MHz`

### Command 5: Logical Disk Space Inspection
- **Command:** `Get-CimInstance Win32_LogicalDisk | Where-Object DriveType -eq 3 | Select-Object DeviceID, Size, FreeSpace`
- **Purpose:** Check storage capacity and free space on primary drives.
- **Exit Code:** `0` (Success)
- **Verified Output:**
  - `C:`: 255.1 GB total, 91.5 GB free (~85.2 GB available)
  - `D:`: 1,000.2 GB total, 618.7 GB free (~576.2 GB available)

### Command 6: GPU / Video Controller Inspection
- **Command:** `Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor`
- **Purpose:** Detect dedicated GPU hardware or CUDA accelerator availability.
- **Exit Code:** `0` (Success)
- **Verified Output:**
  - Name: `Intel(R) HD Graphics`
  - AdapterRAM: `1,073,741,824 bytes` (1 GB shared)
  - CUDA / Discrete GPU: **None detected**. Execution is strictly CPU-only.

### Command 7: Git and Python Runtimes Discovery
- **Command:** `git --version; where.exe python; python --version; py --list`
- **Purpose:** Check Git client and available Python installations.
- **Exit Code:** `0` (Success)
- **Verified Output:**
  - Git: `git version 2.50.1.windows.1`
  - Python installations: `C:\Python314\python.exe`, `Python313`, `Python312`, `Python311`
  - Default Python: `Python 3.14.3`
  - Available targets for PyTorch CPU: Python 3.11 and Python 3.12 verified.

### Command 8: Package Environment Audit in Python 3.11 / 3.12
- **Command:** `py -3.11 -m pip list; py -3.12 -m pip list`
- **Purpose:** Verify existing packages without modifying or installing new libraries.
- **Exit Code:** `0` (Success via Background Task `task-17`)
- **Verified Output:** Both Python 3.11 and 3.12 base installations contain only standard `pip` and `setuptools`. No heavy environments pre-exist.

### Command 9: Workspace Directory Hierarchy Initialization
- **Command:**
  ```powershell
  New-Item -ItemType Directory -Path "pathmnist-federated-learning\third_party" -Force;
  New-Item -ItemType Directory -Path "pathmnist-federated-learning\reports" -Force;
  New-Item -ItemType Directory -Path "pathmnist-federated-learning\research" -Force;
  New-Item -ItemType Directory -Path "pathmnist-federated-learning\experiments" -Force;
  New-Item -ItemType Directory -Path "pathmnist-federated-learning\documentation" -Force;
  ```
- **Purpose:** Create clean, isolated workspace structure.
- **Exit Code:** `0` (Success)

### Command 10: Official PFLlib Repository Acquisition via Git
- **Command:** `git clone https://github.com/TsingZ0/PFLlib.git "pathmnist-federated-learning\third_party\PFLlib"`
- **Purpose:** Clone untouched upstream repository directly from official GitHub source.
- **Exit Code:** `0` (Success via Background Task `task-26`)
- **Result:** Successfully cloned 196 files into `pathmnist-federated-learning\third_party\PFLlib`.

### Command 11: Repository Metadata Verification
- **Command:**
  ```powershell
  cd "pathmnist-federated-learning\third_party\PFLlib"
  git remote -v
  git branch --show-current
  git rev-parse HEAD
  git status --short
  git log -1 --stat
  ```
- **Purpose:** Verify upstream branch, commit SHA, remote URL, and clean working tree.
- **Exit Code:** `0` (Success)
- **Verified Metadata:**
  - Remote: `https://github.com/TsingZ0/PFLlib.git`
  - Default Branch: `master`
  - Checked-out Branch: `master`
  - Commit SHA: `0169ba7e412c9856a08bb3faefab1e35f538a3c1`
  - Commit Date: `Sun Jan 25 10:31:48 2026 +0800`
  - Working Tree: Completely clean (`git status` returned 0 changes).

### Command 12: Repository File Inventory & Classification
- **Command:** Automated PowerShell script counting tracked files by category.
- **Purpose:** Precise quantification of repository assets.
- **Exit Code:** `0` (Success)
- **Inventory Metrics:**
  - Total tracked files: 196
  - Python files: 122
  - Server files: 41
  - Client files: 40
  - Dataset generators: 24
  - Optimizer files: 1
  - Trainmodel files: 6
  - Total physical size: 151.31 MB

---

## 3. Verification Summary

| Check Item | Requirement | Verification Status | Evidence |
|---|---|---|---|
| Upstream Repository Clone | Cloned via official URL | **VERIFIED** | Task 26 completed, exit code 0 |
| Default Branch Check | Identified without assumption | **VERIFIED** | Branch is `master` (not `main`) |
| Git Working Tree Status | Untouched and clean | **VERIFIED** | `git status --short` output is empty |
| Hardware Telemetry | Physical RAM, CPU, GPU | **VERIFIED** | 3.92 GB RAM, 2-Core G3240, Intel HD Graphics |
| Memory Feasibility Analysis | 4 GB RAM budgeting | **VERIFIED** | Report 4 complete; lazy uint8 loading specified |
| FedAvg Training Loop | Traced with file paths | **VERIFIED** | Report 2 complete; `serveravg.py` and `clientavg.py` audited |
| FedProx Formula Verification | $\mu (w - w_t)$ gradient step | **VERIFIED** | Report 2 complete; `PerturbedGradientDescent` verified |
| Upstream Data Leakage Flaw | Train/test merge identified | **VERIFIED** | Report 3 complete; `generate_MNIST.py:53-61` audited |
| All 8 Reports Written | Under `reports/` | **VERIFIED** | All 8 Markdown artifacts generated |
