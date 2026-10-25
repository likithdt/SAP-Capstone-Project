"""
Standard Operating Procedures (SOP) and Equipment Maintenance Manuals Knowledge Base.
Provides grounded reference documentation for RAG vector retrieval in rotating asset maintenance.
"""

from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class MaintenanceManualDocument:
    """Represents a chunked maintenance manual or SOP document."""
    doc_id: str
    title: str
    category: str
    fault_class: int
    fault_name: str
    sop_code: str
    estimated_hours: float
    recommended_parts: List[Dict[str, Any]]
    content: str
    metadata: Dict[str, Any]


EQUIPMENT_MAINTENANCE_MANUALS: List[MaintenanceManualDocument] = [
    MaintenanceManualDocument(
        doc_id="MAN-SOP-MEC-001",
        title="Standard Operating Procedure: Bearing Wear Diagnosis and SKF 6205-2RS Replacement",
        category="Mechanical - Bearings",
        fault_class=1,
        fault_name="Bearing Wear (BPFO)",
        sop_code="SOP-MEC-001",
        estimated_hours=2.5,
        recommended_parts=[
            {
                "material_id": "MAT-BRG-6205-2RS",
                "material_name": "Deep Groove Ball Bearing SKF 6205-2RS (25x52x15mm)",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": True,
            },
            {
                "material_id": "MAT-LUB-LGMT2",
                "material_name": "SKF LGMT 2 General Purpose Industrial Bearing Grease 400g Cartridge",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": False,
            },
        ],
        content="""
SOP-MEC-001: Rolling Element Bearing Wear (BPFO) Diagnosis and Replacement Procedure.
Asset Type: Conveyor Drive Motor (1800 RPM nominal, N=30 Hz fundamental).
Applicable Defect: Ball Pass Frequency Outer Raceway (BPFO) degradation at 3.58 * f0 = 107.4 Hz.

1. Diagnostic Signatures and Thresholds:
- Vibration Kurtosis > 4.5 signals periodic impact shocks from rolling balls striking raceway micro-spalls.
- Crest Factor > 4.0 indicates sharp impulsive acceleration spikes before broadband noise elevation.
- Prominent peak in the demodulated envelope spectrum at 107.4 Hz and its harmonics (214.8 Hz, 322.2 Hz).
- RMS vibration exceeding ISO 10816-3 Class II Alert limit of 0.35 g indicates Class C / D operational zone.

2. Safety and Isolation (LOTO):
- Execute Lockout/Tagout on 3-phase 415V/50Hz supply isolator switch.
- Verify zero mechanical torque stored on conveyor belt drive pulley.

3. Dismantling and Inspection:
- Disconnect drive coupling from motor shaft using hydraulic puller tool P-102.
- Remove end-bell bearing housing cover screws.
- Inspect bearing cavity for grease saponification, contamination, or thermal discoloration.
- Inspect outer raceway fluting or micro-spalling using 10x optical loupe.

4. Replacement Procedure:
- Use induction heater IH-20 to heat new SKF 6205-2RS bearing uniformly to 110 °C. Never use open flame.
- Press fit the heated bearing smoothly onto shaft shoulder until firmly seated against locating collar.
- Pack bearing cavity with 30% fill of SKF LGMT 2 mineral oil-based lithium soap grease.
- Reassemble end-bell housing torquing casing bolts to 28 Nm in cross-pattern.
- Perform hand spin test to verify zero binding or axial rubbing before energizing.
        """,
        metadata={
            "sop_code": "SOP-MEC-001",
            "fault_class": 1,
            "fault_name": "Bearing Wear (BPFO)",
            "work_center": "MECH-01",
            "criticality": "HIGH",
            "primary_material": "MAT-BRG-6205-2RS",
        },
    ),
    MaintenanceManualDocument(
        doc_id="MAN-SOP-MEC-002",
        title="Standard Operating Procedure: Rotor Dynamic Imbalance Diagnosis and Precision Balancing",
        category="Mechanical - Rotor Dynamics",
        fault_class=2,
        fault_name="Rotor Imbalance (1X)",
        sop_code="SOP-MEC-002",
        estimated_hours=1.5,
        recommended_parts=[
            {
                "material_id": "MAT-WGT-BAL-10G",
                "material_name": "Rotor Dynamic Balancing Counterweight Clip Kit (10g / 20g)",
                "quantity_required": 1,
                "unit_of_measure": "SET",
                "is_critical": True,
            }
        ],
        content="""
SOP-MEC-002: Rotor Dynamic Imbalance Diagnostic and Dynamic Field Balancing.
Asset Type: Conveyor Drive Motor (1800 RPM nominal, fundamental f0 = 30.0 Hz).
Applicable Defect: Mass eccentricity causing centrifugal unbalance force proportional to omega squared.

1. Diagnostic Signatures and Thresholds:
- Strong sinusoidal 1X vibration peak strictly at 30.0 Hz in radial acceleration spectrum.
- 1X harmonic power ratio > 70% of total spectral vibration energy.
- Radial RMS vibration elevated (> 0.70 g), while axial vibration remains low (< 0.25 g).
- Vibration phase angle remains stable and 90 degrees out of phase across orthogonal radial sensors.

2. Safety and Field Balancing Setup:
- Implement LOTO during sensor and reflective tape placement.
- Clean rotor fan blades and coupling hub; check for debris accumulation or missing balance weights.
- Mount optical tachometer probe aligned with reflective target tape on motor shaft.

3. Balancing Procedure (ISO 1940-1 Grade G2.5):
- Execute baseline run to record initial 1X amplitude (g) and phase angle (degrees).
- Place trial weight (10g) at known angular location on rotor balance ring.
- Run machine, measure change in 1X response vector, and calculate influence coefficient.
- Calculate required correction weight mass and exact phase angle location.
- Install permanent balance clip MAT-WGT-BAL-10G and lock with Loctite 243.
- Run verification trial: 1X vibration must fall below ISO Grade G2.5 limit (0.28 g RMS).
        """,
        metadata={
            "sop_code": "SOP-MEC-002",
            "fault_class": 2,
            "fault_name": "Rotor Imbalance (1X)",
            "work_center": "MECH-01",
            "criticality": "MEDIUM",
            "primary_material": "MAT-WGT-BAL-10G",
        },
    ),
    MaintenanceManualDocument(
        doc_id="MAN-SOP-MEC-003",
        title="Standard Operating Procedure: Shaft Misalignment Correction and Laser Coupling Realignment",
        category="Mechanical - Shaft & Coupling",
        fault_class=3,
        fault_name="Shaft Misalignment (2X/3X)",
        sop_code="SOP-MEC-003",
        estimated_hours=3.0,
        recommended_parts=[
            {
                "material_id": "MAT-SHM-SS-KIT",
                "material_name": "Stainless Steel Precision Slotted Machinery Shims Assortment Kit (0.05 - 2.0mm)",
                "quantity_required": 1,
                "unit_of_measure": "SET",
                "is_critical": True,
            },
            {
                "material_id": "MAT-CPL-SPIDER-01",
                "material_name": "Elastomeric Spider Insert for Flexible Jaw Coupling HRC-70",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": False,
            },
        ],
        content="""
SOP-MEC-003: Drive Shaft Misalignment Diagnosis and Precision Laser Realignment.
Asset Type: Conveyor Drive Motor connected to Conveyor Head Pulley via Flexible Coupling.
Applicable Defect: Angular and parallel offset misalignment between motor shaft and gearbox input.

1. Diagnostic Signatures and Thresholds:
- Pronounced 2X (60 Hz) and 3X (90 Hz) harmonic peaks in FFT acceleration spectrum.
- 2X harmonic amplitude exceeding 50% of fundamental 1X amplitude.
- Axial-to-radial vibration RMS ratio > 0.55, reflecting high cyclic bending stress across coupling.
- Temperature rise at bearing housing adjacent to drive-end coupling.

2. Safety and Coupling Inspection:
- Execute electrical LOTO on motor drive control cabinet.
- Remove coupling guard and inspect elastomeric spider insert (MAT-CPL-SPIDER-01) for black dust or teeth wear.
- Check soft foot condition on motor support base using 0.05 mm feeler gauge; correct if soft foot > 0.05 mm.

3. Laser Alignment Procedure:
- Mount laser transmitter and detector fixtures on motor and driven shaft hubs.
- Rotate shafts simultaneously through 90, 180, and 270 degrees.
- Measure vertical and horizontal angular gap and parallel offset.
- Target tolerances: Angular misalignment < 0.05 mm/100mm, Parallel offset < 0.05 mm.
- Add or remove precision stainless steel shims (MAT-SHM-SS-KIT) under motor foot pads.
- Tighten foundation bolts in cross pattern to 75 Nm and execute final verification sweep.
        """,
        metadata={
            "sop_code": "SOP-MEC-003",
            "fault_class": 3,
            "fault_name": "Shaft Misalignment (2X/3X)",
            "work_center": "MECH-01",
            "criticality": "MEDIUM",
            "primary_material": "MAT-SHM-SS-KIT",
        },
    ),
    MaintenanceManualDocument(
        doc_id="MAN-SOP-ELE-001",
        title="Standard Operating Procedure: Stator Thermal Overheating and Cooling System Overhaul",
        category="Electrical - Thermal Management",
        fault_class=4,
        fault_name="Thermal Overheating",
        sop_code="SOP-ELE-001",
        estimated_hours=2.0,
        recommended_parts=[
            {
                "material_id": "MAT-FAN-COOL-100",
                "material_name": "Electric Motor External Cooling Fan Impeller D100mm Plastic",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": True,
            },
            {
                "material_id": "MAT-THM-PST-50G",
                "material_name": "High-Conductivity Heat Sink Thermal Compound Paste (50g)",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": False,
            },
        ],
        content="""
SOP-ELE-001: Electric Motor Thermal Overheating Diagnosis and Cooling Circuit Restoration.
Asset Type: 3-Phase Induction Drive Motor (Class F Insulation, Max Allowable 155 °C, Operational Limit 85 °C).
Applicable Defect: Stator winding or bearing housing temperature exceeding Class Threshold 85.0 °C.

1. Diagnostic Signatures and Thresholds:
- Motor skin/bearing temperature exceeding 80.0 °C (Warning) or 85.0 °C (Critical).
- Rate of temperature rise (dT/dt) exceeding +0.5 °C/min under steady-state mechanical load.
- Thermal imaging indicates localized hotspot at non-drive end cowl or stator core center.

2. Safety and Thermal Isolation:
- Allow motor casing to cool below 45 °C before touching or disassembling fan shroud.
- Execute LOTO on electrical control gear.

3. Root Cause Investigation:
- Inspect cooling fan cowl for lint, dust buildup, or mechanical blockage choking airflow.
- Inspect external cooling fan plastic impeller (MAT-FAN-COOL-100) for cracked, missing, or stripped blades.
- Test stator insulation resistance using 1000V Megohmmeter (acceptable if > 10 Megohms).
- Measure winding DC resistance across U-V, V-W, W-U phases (must balance within 2%).

4. Corrective Maintenance Steps:
- Clean motor stator cooling ribs using compressed air and industrial degreaser.
- Replace damaged cooling fan impeller with MAT-FAN-COOL-100; secure retaining snap-ring.
- Reapply thermal paste MAT-THM-PST-50G on internal temperature sensor probe junction.
- Check ambient ventilation fan in motor cubicle.
- Re-energize and verify steady-state temperature stabilizes below 52.0 °C nominal.
        """,
        metadata={
            "sop_code": "SOP-ELE-001",
            "fault_class": 4,
            "fault_name": "Thermal Overheating",
            "work_center": "ELEC-01",
            "criticality": "HIGH",
            "primary_material": "MAT-FAN-COOL-100",
        },
    ),
    MaintenanceManualDocument(
        doc_id="MAN-SOP-MEC-004",
        title="Standard Operating Procedure: Motor Overload Diagnosis and Conveyor Mechanical Drive Relief",
        category="Mechanical / Electrical - Overload",
        fault_class=5,
        fault_name="Motor Overload",
        sop_code="SOP-MEC-004",
        estimated_hours=1.0,
        recommended_parts=[
            {
                "material_id": "MAT-BELT-V-A42",
                "material_name": "Heavy Duty Industrial V-Belt Section A42 (Conveyor Drive)",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": True,
            }
        ],
        content="""
SOP-MEC-004: Motor Overload and Conveyor Mechanical Binding Diagnostic Procedure.
Asset Type: Conveyor Drive Motor with reduction gearbox and belt transmission.
Applicable Defect: Elevated current draw > 2.4 A (Nominal 1.1-1.3 A) with 100 Hz supply ripple.

1. Diagnostic Signatures and Thresholds:
- Average phase current continuously exceeding 2.40 A (up to 3.80 A stall region).
- 100 Hz current ripple (2 * f_line) reflecting severe electrical and magnetic slip under load.
- Motor speed sagging below 1740 RPM (nominal 1800 RPM synchronous speed).
- Elevated thermal drift accompanying prolonged overcurrent draw.

2. Safety and Immediate Mitigation:
- Immediately pause upstream bulk material feeding into conveyor belt.
- Switch motor drive controller to manual jog mode.

3. Mechanical Investigation Steps:
- Inspect conveyor rollers for foreign material jamming, scrap wrap, or seized idler bearings.
- Inspect conveyor belt tension and track alignment; adjust take-up screw if belt is over-tensioned.
- Inspect drive V-belt (MAT-BELT-V-A42) for glazing, bottoming out in pulley grooves, or severe slip.
- Check gearbox oil level and viscosity; verify free rotation by hand in neutral clutch position.

4. Action Plan:
- Clear all mechanical obstructions from conveyor bed.
- Replace worn drive belt with MAT-BELT-V-A42 and adjust center distance to 12 mm deflection under 20 N.
- Resume conveyor operation at 50% throughput for 15 minutes while monitoring current draw (< 1.30 A).
        """,
        metadata={
            "sop_code": "SOP-MEC-004",
            "fault_class": 5,
            "fault_name": "Motor Overload",
            "work_center": "MECH-01",
            "criticality": "HIGH",
            "primary_material": "MAT-BELT-V-A42",
        },
    ),
    MaintenanceManualDocument(
        doc_id="MAN-SOP-INS-001",
        title="Standard Operating Procedure: Sensor Anomaly Diagnosis, Wiring Check, and Transducer Replacement",
        category="Instrumentation - Sensors",
        fault_class=6,
        fault_name="Sensor Anomaly",
        sop_code="SOP-INS-001",
        estimated_hours=0.5,
        recommended_parts=[
            {
                "material_id": "MAT-SENS-MPU6050",
                "material_name": "6-Axis Digital Accelerometer and Gyroscope Sensor Module MPU-6050",
                "quantity_required": 1,
                "unit_of_measure": "EA",
                "is_critical": True,
            }
        ],
        content="""
SOP-INS-001: Instrumentation Sensor Glitch, Signal Dropout, and Transducer Replacement.
Asset Type: Industrial IoT Telemetry Node (ESP32 / Pico W with MPU-6050, DS18B20, ACS712).
Applicable Defect: Unphysical spikes (> 15 g), NaN values, signal clipping, or telemetry loss.

1. Diagnostic Signatures and Thresholds:
- Vibration acceleration exceeding +/- 15.0 g with zero accompanying thermal or current elevation.
- Zero-variance flatline signal indicating sensor disconnection or frozen I2C bus.
- Temperature reading reporting -127.0 °C (1-wire disconnection) or 85.0 °C power-on reset state.
- ACS712 current reading floating around zero or max ADC rail.

2. Inspection Procedure:
- Inspect physical mounting stud of MPU-6050 sensor on motor bearing end-plate. Mounting torque must be 2.5 Nm.
- Check 4-conductor shielded cable for pinch points, abrasion, or ground loop noise.
- Verify 3.3V DC power rail stability and pull-up resistors (4.7kΩ on I2C SDA/SCL and 1-wire bus).

3. Corrective Action:
- Power-cycle edge microcontroller to clear I2C bus lockup.
- If unphysical spikes persist, replace damaged MPU-6050 sensor with MAT-SENS-MPU6050.
- Calibrate accelerometer zero-g offset before resuming automated digital twin telemetry stream.
        """,
        metadata={
            "sop_code": "SOP-INS-001",
            "fault_class": 6,
            "fault_name": "Sensor Anomaly",
            "work_center": "ELEC-01",
            "criticality": "LOW",
            "primary_material": "MAT-SENS-MPU6050",
        },
    ),
    MaintenanceManualDocument(
        doc_id="MAN-SOP-GEN-001",
        title="Standard Operating Procedure: Conveyor Drive Motor Nominal Baseline and Routine PM Protocol",
        category="General - Preventive Maintenance",
        fault_class=0,
        fault_name="Healthy",
        sop_code="SOP-GEN-001",
        estimated_hours=0.5,
        recommended_parts=[],
        content="""
SOP-GEN-001: Conveyor Drive Motor Nominal Operating Specifications and Routine Preventive Maintenance.
Asset Type: Conveyor Drive Motor (MOTOR-001, 1800 RPM, 3-Phase 415V).
Applicable State: Nominal Healthy Operation (Class 0).

1. Baseline Operating Ranges:
- Vibration RMS: 0.15 g to 0.35 g (ISO 10816-3 Class II Zone A/B).
- Operating Temperature: 45.0 °C to 52.0 °C under continuous rated conveyor load.
- Motor Current: 1.10 A to 1.30 A balanced across all three phases.
- Speed: 1780 to 1800 RPM nominal.

2. Routine Maintenance Schedule:
- Monthly visual check of terminal box, ground wire, and foundation bolt torque.
- 500 operating hours: re-lubrication check on bearing grease relief ports.
- Quarterly infrared thermographic survey of stator casing and coupling.
- No corrective maintenance or work order required during nominal healthy operation.
        """,
        metadata={
            "sop_code": "SOP-GEN-001",
            "fault_class": 0,
            "fault_name": "Healthy",
            "work_center": "MECH-01",
            "criticality": "NONE",
            "primary_material": "NONE",
        },
    ),
]
