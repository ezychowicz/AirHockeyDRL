push!(LOAD_PATH, joinpath(@__DIR__, "..", "src")) # zeby dzialalo jakby bylo w src
using Plots, Environment, CoreTypes
gr()  # backend do animacji
 
using Environment
# === INICJALIZACJA === #
params = Environment.EnvParams(
    x_len = 100.0f0,
    y_len = 50.0f0,
    goal_width = 20.0f0,
    puck_radius = 1.0f0,
    mallet_radius = 2.0f0,
    agent1_initial_pos = [10.0f0, 25.0f0],
    agent2_initial_pos = [90.0f0, 25.0f0],
    dt = 0.2f0,
    terminal_velo = 1.0f0,
    band_e_loss = 0.05f0,
    restitution = 0.95f0,
    puck_mass = 1.0f0,
    mallet_mass = 2.0f0
)

env = Environment.AirHockeyEnv(params, Environment.State(nothing, nothing, Environment.Puck([0.0f0, 0.0f0], [0.0f0, 0.0f0])), false)
Environment.reset!(env)


env.state.puck.v .= [60.0f0, 15.0f0]

# === ANIMACJA === #
frames = @gif for i in 1:60
    Environment.simulate_dt!(env)
    
    puck_pos = env.state.puck.pos
    scatter(
        [puck_pos[1]], [puck_pos[2]];
        xlim=(0, params.x_len), ylim=(0, params.y_len),
        legend=false, xlabel="x", ylabel="y",
        title="Klatka $i",
        markersize=8, markercolor=:red
    )

    # opcjonalnie: zaznacz bandy, bramki, itp.
end every 1

