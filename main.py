from network_graph import G
from evacuation_sim import EvacuationSim

# If running inline, ensure G is the projected graph (EPSG:3826) with 'x','y' node attributes.
try:
    G  # noqa: F821
except NameError:
    raise RuntimeError("This example expects a variable `G` (projected graph) in the namespace.")

# create sim
sim = EvacuationSim(
    G=G,
    base_speed_m_s=1.5,
    default_capacity=3,  # tune to model narrow streets
    alpha=1.0,
    beta=2.0,
    time_step=1.0,
    random_seed=28,
)

# spawn agents
# start_positions = [
# (121.43502733028838, 25.175015301017766),  # lon, lat
# (121.44055819479243, 25.171712505525672),
# ]
# sim.spawn_agents(start_positions=start_positions)
sim.spawn_agents(n_agents=50)

# run simulation up to max_time_s seconds
sim.run(max_time_s=1800, verbose=True)

# export CSV for Kepler.gl (lon/lat)
df = sim.export_records_to_csv("output_files/agent_positions.csv", to_wgs84=True)

print("Done.")