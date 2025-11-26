import numpy as np
import pandas as pd
from pprint import pprint
from app.model import Model
from app.subcatchmentGraph import SubcatchmentGraph

# Setup
file = "largerExample"
tempRainfall = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0])
rainInfo = {
    "spaceConversion": 0.0254,
    "timeConversion": 3600,
    "rainfall": tempRainfall,
    "rainfallTimes": np.array([i for i in range(len(tempRainfall))]),
}
dt = 1800

model = Model(file, dt, rainInfo, oldwaterRatio=0.2)

print("=" * 60)
print("DIAGNOSTIC 1: Subcatchment Setup")
print("=" * 60)
print(f"Number of subcatchments: {model.subcatchment.G.vcount()}")
print(f"Subcatchment areas: {model.subcatchment.G.vs['area']}")
print(f"Subcatchment widths: {model.subcatchment.G.vs['width']}")
print(f"Subcatchment slopes: {model.subcatchment.G.vs['slope']}")
print(f"hydraulicCoupling (outgoing IDs): {model.subcatchment.hydraulicCoupling}")
print(f"oldwaterRatio: {model.subcatchment.oldwaterRatio}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Check Rainfall Values")
print("=" * 60)
print(f"Normalized rainfall (m/s): {model.rainfall[:5]}...")
print(f"Interpolated rain at timesteps: {model.rain[:5]}...")
print(f"Total subcatchment area: {sum(model.subcatchment.G.vs['area'])} m²")
total_effective_rainfall = sum(model.rain) * dt * sum(model.subcatchment.G.vs['area']) * (1 - 0.2)
print(f"Expected total effective rainfall volume: {total_effective_rainfall:.2f} m³")

print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Run ONE timestep and check runoff")
print("=" * 60)
# Run one step with significant rainfall
test_rain = 0.001  # m/s (high rainfall)
depths_before = model.subcatchment.G.vs["depth"].copy()
depths, runoffs = model.subcatchment.update(0, dt, model.rain[4])  # Use a mid-storm timestep
print(f"Rain intensity used: {model.rain[4]:.6e} m/s")
print(f"Depths after update: {depths}")
print(f"Runoffs (m³/s): {runoffs}")
print(f"Total runoff (m³/s): {sum(runoffs):.6e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 4: Check coupling array indexing")
print("=" * 60)
print(f"Coupling array size: {len(model.coupling['subcatchmentRunoff'])}")
print(f"hydraulicCoupling values: {model.subcatchment.hydraulicCoupling}")
print(f"hydraulicCoupling - 1 (0-based): {model.subcatchment.hydraulicCoupling - 1}")

# Manually set runoff and check mapping
for v in model.subcatchment.G.vs:
    v["runoff"] = 1.0  # Set to 1.0 for easy checking
model.updateRunoff()
print(f"After updateRunoff with runoff=1.0:")
print(f"subcatchmentRunoff array: {model.coupling['subcatchmentRunoff']}")
print(f"Non-zero indices: {np.where(model.coupling['subcatchmentRunoff'] > 0)[0]}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 5: Check street node coupledIDs")
print("=" * 60)
print("Street nodes that should receive runoff:")
for v in model.street.G.vs:
    print(f"  Node {v.index}: coupledID={v['coupledID']}, reads from coupling[{v['coupledID']-1}]")

print("\n" + "=" * 60)
print("DIAGNOSTIC 6: Full simulation mass balance")
print("=" * 60)
# Reset model
model = Model(file, dt, rainInfo, oldwaterRatio=0.2)
model.run()

total_street_outflow = sum(model.streetOutfallFlows) * dt
total_sewer_outflow = sum(model.sewerOutfallFlows) * dt
total_outflow = total_street_outflow + total_sewer_outflow

# Recalculate expected
total_area = sum(model.subcatchment.G.vs['area'])
total_rain_volume = sum(model.rain) * dt * total_area
effective_rain_volume = total_rain_volume * (1 - model.subcatchment.oldwaterRatio)

print(f"Total rain volume: {total_rain_volume:.2f} m³")
print(f"Effective rain volume (after oldwater): {effective_rain_volume:.2f} m³")
print(f"Total street outflow: {total_street_outflow:.2f} m³")
print(f"Total sewer outflow: {total_sewer_outflow:.2f} m³")
print(f"Total outflow: {total_outflow:.2f} m³")
print(f"Mass balance ratio: {total_outflow/effective_rain_volume:.2%}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 7: Check if runoff reaches street network")
print("=" * 60)
# Reset and run step by step
model = Model(file, dt, rainInfo, oldwaterRatio=0.2)
for n in range(3):  # Just first 3 steps
    model.step(n)
    total_runoff = sum(v["runoff"] for v in model.subcatchment.G.vs)
    coupling_sum = sum(model.coupling["subcatchmentRunoff"])
    print(f"Step {n}: subcatchment total runoff={total_runoff:.6e}, coupling sum={coupling_sum:.6e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 8: Check qFull capacity vs actual inflows")
print("=" * 60)
model = Model(file, dt, rainInfo, oldwaterRatio=0.2)

# Check street network capacity
print("Street network edge capacities (qFull):")
for e in model.street.G.es:
    print(f"  Edge {e.index}: qFull={e['qFull']:.6f} m³/s, slope={e['slope']:.4f}")

# Run a few steps and check flows
for n in range(5):
    model.step(n)

print(f"\nAfter 5 steps, street edge Q1 values:")
for e in model.street.G.es:
    print(f"  Edge {e.index}: Q1={e['Q1']:.6f}, Q2={e['Q2']:.6f}, qFull={e['qFull']:.6f}, Q1/qFull={e['Q1']/e['qFull']:.2%}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 9: Trace water through network step by step")
print("=" * 60)
model = Model(file, dt, rainInfo, oldwaterRatio=0.2)

for n in range(3):
    # Before step
    coupling_before = model.coupling["subcatchmentRunoff"].copy()
    
    model.step(n)
    
    # Check what entered vs what's in pipes vs what exited
    total_Q1 = sum(e["Q1"] for e in model.street.G.es)
    total_Q2 = sum(e["Q2"] for e in model.street.G.es)
    street_outfall = model.streetOutfallFlows[-1] if model.streetOutfallFlows else 0
    coupling_sum = sum(model.coupling["subcatchmentRunoff"])
    
    print(f"Step {n}:")
    print(f"  Coupling sum (inflow): {coupling_sum:.6f} m³/s")
    print(f"  Total Q1 in pipes: {total_Q1:.6f} m³/s")
    print(f"  Total Q2 in pipes: {total_Q2:.6f} m³/s")
    print(f"  Street outfall: {street_outfall:.6f} m³/s")

print("\n" + "=" * 60)
print("DIAGNOSTIC 10: Check getIncomingCoupled for specific nodes")
print("=" * 60)
model = Model(file, dt, rainInfo, oldwaterRatio=0.2)

# Set coupling manually
model.coupling["subcatchmentRunoff"] = np.zeros(39)
model.coupling["subcatchmentRunoff"][4] = 0.05  # Should go to street node with coupledID=5

print("Set coupling[4] = 0.05")
print(f"Street node 3 has coupledID={model.street.G.vs[3]['coupledID']}")
print(f"Checking coupling[coupledID-1] = coupling[{model.street.G.vs[3]['coupledID']-1}] = {model.coupling['subcatchmentRunoff'][model.street.G.vs[3]['coupledID']-1]}")

# Now check what getIncomingCoupled would return
for v in model.street.G.vs[:5]:
    idx = v["coupledID"] - 1
    runoff = model.coupling["subcatchmentRunoff"][idx]
    drain = model.coupling["drainCapture"][idx]
    overflow = model.coupling["drainOverflow"][idx]
    total = runoff + drain + overflow
    print(f"  Node {v.index} (coupledID={v['coupledID']}): runoff={runoff:.4f}, drain={drain:.4f}, overflow={overflow:.4f}, total={total:.4f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 11: Check if water is stuck in storage")
print("=" * 60)
model = Model(file, dt, rainInfo, oldwaterRatio=0.2)
model.run()

# Check final state of pipes
total_A1 = sum(e["A1"] for e in model.street.G.es)
total_A2 = sum(e["A2"] for e in model.street.G.es)
total_length = sum(e["length"] for e in model.street.G.es)

# Estimate water stored in pipes (rough: avg area × length)
avg_area = (total_A1 + total_A2) / (2 * model.street.G.ecount())
total_pipe_length = sum(e["length"] for e in model.street.G.es)
stored_volume_estimate = sum((e["A1"] + e["A2"]) / 2 * e["length"] for e in model.street.G.es)

print(f"Final pipe storage estimate: {stored_volume_estimate:.2f} m³")
print(f"Total pipe length: {total_pipe_length:.2f} m")

# Check subcatchment final depths
final_subcatch_storage = sum(d * a for d, a in zip(model.subcatchment.G.vs["depth"], model.subcatchment.G.vs["area"]))
print(f"Final subcatchment storage: {final_subcatch_storage:.2f} m³")

print("\n" + "=" * 60)
print("DIAGNOSTIC 12: Verify order of operations in step()")
print("=" * 60)
print("Current order in step():")
print("  1. subcatchment.update() - generates runoff")
print("  2. street.update(self.coupling) - uses PREVIOUS coupling values!")
print("  3. sewer.update(self.coupling)")
print("  4. updateDrainCapture()")
print("  5. updateRunoff() - updates coupling for NEXT timestep")
print("")
print("BUG: Street network uses coupling from PREVIOUS timestep!")
print("First timestep uses all-zero coupling (no runoff reaches street)")
