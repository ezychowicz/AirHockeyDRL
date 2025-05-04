# Dodaj ścieżkę do źródeł, jeśli jest potrzebne
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Importujemy potrzebne moduły
using Plots
using AirHockey  
include("TracedEnv.jl")
using .TracedEnv
gr()  # Ustawienie backendu dla animacji

# === INICJALIZACJA === #
params = AirHockey.EnvParams(  
    x_len = 100.0f0,
    y_len = 50.0f0,
    goal_width = 20.0f0,
    puck_radius = 1.0f0,
    mallet_radius = 2.0f0,
    agent1_initial_pos = [10.0f0, 25.0f0],
    agent2_initial_pos = [90.0f0, 25.0f0],
    dt = 0.02f0,
    terminal_velo = 1.0f0,
    band_e_loss = 0.05f0,
    restitution = 0.95f0,
    puck_mass = 1.0f0,
    mallet_mass = 2.0f0
)

# Tworzymy instancję środowiska z AirHockeyEnv
env = AirHockey.AirHockeyEnv(params, AirHockey.State(nothing, nothing, AirHockey.Puck([0.0f0, 0.0f0], [0.0f0, 0.0f0])), false)

# Resetujemy środowisko
AirHockey.reset!(env)

# Ustawiamy początkową prędkość krążka
env.state.puck.v .= [60.0f0, 15.0f0]

# === ANIMACJA === #
frames = @gif for i in 1:50
    trace = TracedEnv.simulate_dt!(env)  

    puck_positions = trace.puck_trace
    println(puck_positions)
    xs, ys = map(x -> x[1], puck_positions), map(x -> x[2], puck_positions)
    
    println(xs,ys)
    scatter(
        xs, ys; 
        xlim=(0, params.x_len), ylim=(0, params.y_len),
        legend=false, xlabel="x", ylabel="y",
        title="Klatka $i",
        markersize=8, markercolor=:red
    )

    # opcjonalnie: dodaj inne elementy jak bandy, bramki itp.
end every 1


