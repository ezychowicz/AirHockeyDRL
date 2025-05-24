# Dodaj ścieżkę do źródeł, jeśli jest potrzebne
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Importujemy potrzebne moduły
using Plots
using GLMakie
# using GeometryBasics
using AirHockey  
include("TracedEnv.jl")
using .TracedEnv
# gr()  # Ustawienie backendu dla animacji

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
    band_e_loss = 0.95f0,
    restitution = 0.95f0,
    puck_mass = 1.0f0,
    mallet_mass = 2.0f0
)

# Tworzymy instancję środowiska z AirHockeyEnv
agent1 = AirHockey.Mallet([0.0f0, 0.0f0], [5.0f0, 25.0f0])
agent2 = AirHockey.Mallet([0.0f0, 0.0f0], [95.0f0, 25.0f0])
env = AirHockey.AirHockeyEnv(params, AirHockey.State(agent1, agent2, AirHockey.Puck([0.0f0, 0.0f0], [25.0f0, 25.0f0])), false)



# Ustawiamy początkową prędkość krążka
env.state.puck.v .= [-5.0f0, 1.0f0]

# === ANIMACJA === #
function interpolate_movement(xs::Vector{V},ys::Vector{V},vs::Vector{Vector{T}}, times::Vector{Float32}; precision::Int64 = 1000) where {V,T <: Real}
    """
    Interpoluj ruchy między kolejnymi stanami uzyskanymi z całej symulacji. 
    Uzupełniaj stanami pośrednimi, zakładając r. jednostajny prostoliniowy.
    vs niepotrzebne przy takim założeniu, ale wole zostawić.
    """
    xs_diffs = xs[2:end] .- xs[1:end-1]
    new_xs, new_ys = Vector{V}(), Vector{V}()
    full_time = sum(times) # czas calej symulacji
    for i in eachindex(xs_diffs)
        k = times[i]/full_time # czas kroku wzgledem danej symulacji: liczba klatek ma być proporcjonalna
        frames = max(2,round(Int64,k*precision))
        println("frames:$frames")
        xs_filler = collect(range(xs[i], xs[i+1], frames))
        ys_filler = collect(range(ys[i], ys[i+1], frames))
        append!(new_xs, xs_filler)
        append!(new_ys, ys_filler)
    end
    new_xs, new_ys
end

function simulate()
    puck_states = Vector{AirHockey.Puck}()
    mallet1_states = Vector{AirHockey.Mallet}()
    mallet2_states = Vector{AirHockey.Mallet}()
    times_cumul = Vector{Float32}()
    for _ ∈ 1:50
        trace = TracedEnv.simulate_dt!(env)  
        append!(times_cumul, trace.times)
        append!(puck_states, trace.puck_trace)
        append!(mallet1_states, trace.mallet1_trace)
        append!(mallet2_states, trace.mallet2_trace)
    end
    puck_states, mallet1_states, mallet2_states, times_cumul
end

function interpolate_simulation(puck_states, mallet1_states, mallet2_states, times)
    puck_positions = map(puck -> puck.pos, puck_states)
    mallet1_positions = map(mallet -> mallet.pos, mallet1_states)
    mallet2_positions = map(mallet -> mallet.pos, mallet2_states)
    puck_vs = map(puck -> puck.v, puck_states)
    mallet1_vs = map(mallet -> mallet.v, mallet1_states)
    mallet2_vs = map(mallet -> mallet.v, mallet2_states)
    puck_xs, puck_ys = map(x -> x[1], puck_positions), map(x -> x[2], puck_positions)
    mallet1_xs, mallet1_ys = map(x -> x[1], mallet1_positions), map(x -> x[2], mallet1_positions)
    mallet2_xs, mallet2_ys = map(x -> x[1], mallet2_positions), map(x -> x[2], mallet2_positions)
    new_puck_xs, new_puck_ys = interpolate_movement(puck_xs,puck_ys,puck_vs, times)
    new_mallet1_xs, new_mallet1_ys = interpolate_movement(mallet1_xs, mallet1_ys, mallet1_vs, times)
    new_mallet2_xs, new_mallet2_ys = interpolate_movement(mallet2_xs, mallet2_ys, mallet2_vs, times)
    return new_puck_xs, new_puck_ys, new_mallet1_xs, new_mallet1_ys, new_mallet2_xs, new_mallet2_ys
end

function one_step_progress(i)
    new_puck_xs[i], new_puck_ys[i], new_mallet1_xs[i], new_mallet1_ys[i], new_mallet2_xs[i], new_mallet2_ys[i]
end 

function animstep!(axis, xy_puck, xy_mallet12, i)
    a,b,c,d,e,f = one_step_progress(i)
    xy_puck[] = [Point2f(a,b)]
    xy_mallet12[] = [Point2f(c,d),Point2f(e,f)] 
    axis.title[] = "Klatka $i"
end


new_puck_xs, new_puck_ys, new_mallet1_xs, new_mallet1_ys, new_mallet2_xs, new_mallet2_ys = interpolate_simulation(simulate()...)
xy_puck = Observable([Point2f(new_puck_xs[1], new_puck_ys[1])])
xy_mallet12 = Observable([Point2f(new_mallet1_xs[1], new_mallet1_ys[1]), Point2f(new_mallet2_xs[1], new_mallet2_ys[1])])
f = Figure()
f = Figure(size = (600, 400),camera = campixel!)
scene = Scene(size = (600, 400),camera = campixel!)
axis = f[1,1] = Axis(scene;
    limits = ((0,params.x_len), (0,params.y_len)),
    xlabel = "x", ylabel = "y",
    title = "Klatka 1"
)

scatter_puck = GLMakie.scatter!(axis, xy_puck; color = :red, markersize = 20env.params.puck_radius)
scatter_mallets = GLMakie.scatter!(axis, xy_mallet12; color = :blue, markersize = 20env.params.mallet_radius)
display(scene)

for i in eachindex(new_puck_xs)
    animstep!(axis, xy_puck, xy_mallet12, i)
    sleep(0.01)
end
