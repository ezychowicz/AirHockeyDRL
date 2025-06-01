
module Visualize
export visualize
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using GLMakie
using AirHockey  
include("TracedEnv.jl")
using .TracedEnv
using StaticArrays
function update_axis!(xy_puck::Observable, xy_mallet12::Observable, score::Observable, rewards::Observable, axis::Axis, score_vec::Vector{Int64}; coords::NTuple{6,V}, rewards_vec::Vector{Float32}, t::Float64) where {V <: Real}
    """
    Aktualizuj Observable i tytuł osi.
    """
    x,y,x_m1,y_m1,x_m2,y_m2 = coords
    xy_puck[] = [Point2f(x, y)]
    xy_mallet12[] = [Point2f(x_m1, y_m1), Point2f(x_m2, y_m2)]
    score[] = score_vec
    rewards[] = [rewards_vec[1], rewards_vec[2]]
    # println("score $(score) a score_vec: $(score_vec)")

    axis.title[] = "Czas: $(round(t, digits=2)) s"
end


function interpolate(idx::Int64, k::Float64, puck_xs::Vector{V}, puck_ys::Vector{V}, mallet1_xs::Vector{V}, mallet1_ys::Vector{V}, mallet2_xs::Vector{V}, mallet2_ys::Vector{V}) where {V<:Real}
    x = puck_xs[idx] + (puck_xs[idx+1] - puck_xs[idx]) * k
    y = puck_ys[idx] + (puck_ys[idx+1] - puck_ys[idx]) * k
    x_m1 = mallet1_xs[idx] + (mallet1_xs[idx+1] - mallet1_xs[idx]) * k
    y_m1 = mallet1_ys[idx] + (mallet1_ys[idx+1] - mallet1_ys[idx]) * k
    x_m2 = mallet2_xs[idx] + (mallet2_xs[idx+1] - mallet2_xs[idx]) * k
    y_m2 = mallet2_ys[idx] + (mallet2_ys[idx+1] - mallet2_ys[idx]) * k
    (x,y,x_m1,y_m1,x_m2,y_m2)
end

function scores(results::Vector{Union{Nothing, Bool}})
    """
    Zamień strumień sygnałów: nothing, true, false na macierz wyników (n x 2). (taki cumsum)
    """
    curr_score = Int64[0,0]
    scores = zeros(Int64, 2, length(results))
    for i in eachindex(results)
        if !isnothing(results[i])
            curr_score .+= [!results[i], results[i]]
        end
        scores[:, i] .= curr_score
    end
    println(scores)
    return scores
    
end


function visualize(params::EnvParams, puck_states::Vector{Puck}, mallet1_states::Vector{Mallet}, mallet2_states::Vector{Mallet}, time_diffs::Vector{Float32}, results::Vector{Union{Nothing, Bool}}, rewards::Vector{Vector{Float64}})
    valid_indices = findall(t -> t > 0, time_diffs)
    puck_states = puck_states[valid_indices]
    mallet1_states = mallet1_states[valid_indices]
    mallet2_states = mallet2_states[valid_indices]
    time_diffs = time_diffs[valid_indices]
    results = results[valid_indices]
    rewards = rewards[valid_indices]

    println("$(length(results)), $(length(puck_states))")
    non_nothings = filter(!isnothing, results)
    println("gole: $(non_nothings)")

    puck_positions = map(puck -> puck.pos, puck_states)
    puck_xs = map(x -> x[1], puck_positions)
    puck_ys = map(x -> x[2], puck_positions)
    mallet1_positions = map(mallet -> mallet.pos, mallet1_states)
    mallet2_positions = map(mallet -> mallet.pos, mallet2_states)
    mallet1_xs, mallet1_ys = map(x -> x[1], mallet1_positions), map(x -> x[2], mallet1_positions)
    mallet2_xs, mallet2_ys = map(x -> x[1], mallet2_positions), map(x -> x[2], mallet2_positions)
    scores_matrix = scores(results)
    mallet1_rewards, mallet2_rewards = map(pair -> pair[1], rewards), map(pair -> pair[2], rewards)

    t_values = cumsum([0.0; time_diffs]) # t_values - realne czasy danych stanów
    total_time = sum(time_diffs)

    # # --- Inicjalizacja wizualizacji --- #
    # f = Figure(size = (600, 400), camera = campixel!)
    # scene = Scene(size = (600, 400), camera = campixel!)
    # axis = f[1,1] = Axis(scene;
    #     limits = ((0, params.x_len), (0, params.y_len)),
    #     xlabel = "x", ylabel = "y",
    #     title = "Czas: 0.00 s"
    # )

    # # --- RYSOWANIE BRAMEK --- #
    # goal_half = params.goal_width / 2
    # center_y = params.y_len / 2
    # goal_ymin = center_y - goal_half
    # goal_ymax = center_y + goal_half


    # lines!(axis, [0.0, 0.0], [goal_ymin, goal_ymax], color = :green, linewidth = 10)

    # lines!(axis, [params.x_len, params.x_len], [goal_ymin, goal_ymax], color = :green, linewidth = 10)

    # # --- OBRAMOWANIE BOISKA --- #
    # x_rect = [0.0, params.x_len, params.x_len, 0.0, 0.0]
    # y_rect = [0.0, 0.0, params.y_len, params.y_len, 0.0]
    # lines!(axis, x_rect, y_rect, color = :black, linewidth = 2)

    # xy_puck = Observable([Point2f(puck_xs[1], puck_ys[1])])
    # xy_mallet12 = Observable([Point2f(mallet1_xs[1], mallet1_ys[1]), Point2f(mallet2_xs[1], mallet2_ys[1])])
    # score = Observable([0, 0])
    # rewards = Observable(Float32[0,0])
    # scatter_puck = GLMakie.scatter!(axis, xy_puck; color = :red, markersize = 20*params.puck_radius)
    # scatter_mallets = GLMakie.scatter!(axis, xy_mallet12; color = :blue, markersize = 20*params.mallet_radius)
    # score_text = GLMakie.text!(axis,  @lift(string($(score)[1], " : ", $(score)[2])),  align = (:center, :center), position = Point2f(params.x_len/2, params.y_len-5), fontsize = 30) # lift sprawia że ten napis nie jest statyczny
    # # Dolny rząd — utwórz osobny pod-gridlayout o dwóch kolumnach
    # inner = GridLayout()
    # label_r_left  = Label(f, @lift("Reward 1: $(round($(rewards)[1], digits=2))"), halign = :left)
    # label_r_right = Label(f, @lift("Reward 2: $(round($(rewards)[2], digits=2))"), halign = :right)
    # inner[1, 1:2] = [label_r_left, label_r_right]
    
    # f.layout[1,1] = scene 
    # f.layout[2,1] = inner

    # display(f)
# --- Inicjalizacja wizualizacji --- #
    f = Figure(size = (600, 400))

    axis = f[1, 1:2] = Axis(f;
        limits = ((0, params.x_len), (0, params.y_len)),
        xlabel = "x", ylabel = "y",
        title = "Czas: 0.00 s"
    )

    # --- BRAMKI i OBRAMOWANIE --- #
    goal_half = params.goal_width / 2
    center_y = params.y_len / 2
    goal_ymin = center_y - goal_half
    goal_ymax = center_y + goal_half

    lines!(axis, [0.0, 0.0], [goal_ymin, goal_ymax], color = :green, linewidth = 10)
    lines!(axis, [params.x_len, params.x_len], [goal_ymin, goal_ymax], color = :green, linewidth = 10)

    x_rect = [0.0, params.x_len, params.x_len, 0.0, 0.0]
    y_rect = [0.0, 0.0, params.y_len, params.y_len, 0.0]
    lines!(axis, x_rect, y_rect, color = :black, linewidth = 2)

    # --- Obiekty do rysowania --- #
    xy_puck = Observable([Point2f(puck_xs[1], puck_ys[1])])
    xy_mallet12 = Observable([Point2f(mallet1_xs[1], mallet1_ys[1]), Point2f(mallet2_xs[1], mallet2_ys[1])])
    score = Observable([0, 0])
    rewards = Observable(Float32[0, 0])

    scatter!(axis, xy_puck; color = :red, markersize = 20 * params.puck_radius)
    scatter!(axis, xy_mallet12; color = :blue, markersize = 20 * params.mallet_radius)

    GLMakie.text!(axis,
        @lift(string($(score)[1], " : ", $(score)[2])),
        align = (:center, :center),
        position = Point2f(params.x_len/2, params.y_len - 5),
        fontsize = 30
    )

    # --- Dolny pasek z labelami --- #
    bottom = f[2,1:2] = GridLayout()
    function reward_color(r)
        v = clamp((r + 1) / 2, 0, 1)
        return RGBf(1 - v, v, 0)
    end

    label_r_left = Label(bottom[1, 1],
        @lift("Reward 1: $(round($(rewards)[1], digits=2))"),
        color = @lift(reward_color($(rewards)[1])),
        halign = :left,
        fontsize = 20
    )

    label_r_right = Label(bottom[1, 2],
        @lift("Reward 2: $(round($(rewards)[2], digits=2))"),
        color = @lift(reward_color($(rewards)[2])),
        halign = :right,
        fontsize = 20
    )
    colsize!(bottom, 1, Relative(1/2))
    colsize!(bottom, 2, Relative(1/2))


    display(f)
        # --- Główna pętla animacji --- #
    start_time = time()
    while true
        elapsed = time() - start_time
        t = elapsed % total_time 

        idx = findlast(t_values .<= t) # znajdź indeks ostatniego miniętego czasu. tak naprawdę godzimy się na pomijanie niektórych stanów gdy nie nadążamy
        
        if idx < length(t_values)
            Δt = t - t_values[idx] # ile minęło od ostatniego stanu o którym mamy info
            segment_time = time_diffs[idx]
            coords = interpolate(idx, Δt/segment_time, puck_xs, puck_ys, mallet1_xs, mallet1_ys, mallet2_xs, mallet2_ys)
            rewards_vec = Float32[mallet1_rewards[idx], mallet2_rewards[idx]]
            update_axis!(xy_puck, xy_mallet12, score, rewards, axis, scores_matrix[:, idx]; coords = coords, rewards_vec = rewards_vec, t = t)
            sleep(1/60)
        end
    end
end

end























# module Visualize
# export visualize#, interpolate_movement, interpolate_simulation, one_step_progress, animstep!
# push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# using GLMakie
# using AirHockey  
# include("TracedEnv.jl")
# using .TracedEnv





# function interpolate_movement(xs::Vector{V},ys::Vector{V},vs::Vector{Vector{T}}, times::Vector{Float32}; precision::Int64 = 5000) where {V,T <: Real}
#     """
#     Interpoluj ruchy między kolejnymi stanami uzyskanymi z całej symulacji. 
#     Uzupełniaj stanami pośrednimi, zakładając r. jednostajny prostoliniowy.
#     vs niepotrzebne przy takim założeniu, ale wole zostawić.
#     """
#     xs_diffs = xs[2:end] .- xs[1:end-1]
#     new_xs, new_ys = Vector{V}(), Vector{V}()
#     full_time = sum(times) # czas calej symulacji\
#     # println("lengths:$(length(xs_diffs)) $(length(times))")
#     for i in eachindex(xs_diffs)
#         # println("time=$(times[i]), time_calc=$(xs_diffs[i]/vs[i][1])")
#         k = times[i]/full_time # czas kroku wzgledem danej symulacji: liczba klatek ma być proporcjonalna
#         frames = max(2,round(Int64,k*precision))
#         # println("frames:$frames")
#         xs_filler = collect(range(xs[i], xs[i+1], frames))
#         ys_filler = collect(range(ys[i], ys[i+1], frames))
#         append!(new_xs, xs_filler)
#         append!(new_ys, ys_filler)
#     end
#     new_xs, new_ys
# end


# function interpolate_simulation(puck_states, mallet1_states, mallet2_states, times)
#     puck_positions = map(puck -> puck.pos, puck_states)
#     mallet1_positions = map(mallet -> mallet.pos, mallet1_states)
#     mallet2_positions = map(mallet -> mallet.pos, mallet2_states)
#     puck_vs = map(puck -> puck.v, puck_states)
#     mallet1_vs = map(mallet -> mallet.v, mallet1_states)
#     mallet2_vs = map(mallet -> mallet.v, mallet2_states)
#     puck_xs, puck_ys = map(x -> x[1], puck_positions), map(x -> x[2], puck_positions)
#     mallet1_xs, mallet1_ys = map(x -> x[1], mallet1_positions), map(x -> x[2], mallet1_positions)
#     mallet2_xs, mallet2_ys = map(x -> x[1], mallet2_positions), map(x -> x[2], mallet2_positions)
#     new_puck_xs, new_puck_ys = interpolate_movement(puck_xs,puck_ys,puck_vs, times)
#     new_mallet1_xs, new_mallet1_ys = interpolate_movement(mallet1_xs, mallet1_ys, mallet1_vs, times)
#     new_mallet2_xs, new_mallet2_ys = interpolate_movement(mallet2_xs, mallet2_ys, mallet2_vs, times)
#     return new_puck_xs, new_puck_ys, new_mallet1_xs, new_mallet1_ys, new_mallet2_xs, new_mallet2_ys
# end


# function animstep!(axis, xy_puck, xy_mallet12, i,new_puck_xs, new_puck_ys, new_mallet1_xs, new_mallet1_ys, new_mallet2_xs, new_mallet2_ys)
#     xy_puck[] = [Point2f(new_puck_xs[i], new_puck_ys[i])]
#     xy_mallet12[] = [Point2f(new_mallet1_xs[i], new_mallet1_ys[i]),Point2f(new_mallet2_xs[i], new_mallet2_ys[i])] 
#     axis.title[] = "Klatka $i"
# end


# function visualize(params::EnvParams, puck_states::Vector{Puck}, mallet1_states::Vector{Mallet}, mallet2_states::Vector{Mallet}, time_diffs::Vector{Float32})
#     new_puck_xs, new_puck_ys, new_mallet1_xs, new_mallet1_ys, new_mallet2_xs, new_mallet2_ys = interpolate_simulation(puck_states::Vector{Puck}, mallet1_states::Vector{Mallet}, mallet2_states::Vector{Mallet}, time_diffs::Vector{Float32})
#     # @show new_puck_xs
#     xy_puck = Observable([Point2f(new_puck_xs[1], new_puck_ys[1])])
#     xy_mallet12 = Observable([Point2f(new_mallet1_xs[1], new_mallet1_ys[1]), Point2f(new_mallet2_xs[1], new_mallet2_ys[1])])
#     f = Figure()
#     f = Figure(size = (600, 400),camera = campixel!)
#     scene = Scene(size = (600, 400),camera = campixel!)
#     axis = f[1,1] = Axis(scene;
#         limits = ((0,params.x_len), (0,params.y_len)),
#         xlabel = "x", ylabel = "y",
#         title = "Klatka 1"
#     )
#     # --- RYSOWANIE BRAMEK --- #
#     goal_half = params.goal_width / 2
#     center_y = params.y_len / 2
#     goal_ymin = center_y - goal_half
#     goal_ymax = center_y + goal_half

#     # Lewa bramka (x=0)
#     lines!(axis, [0.0, 0.0], [goal_ymin, goal_ymax], color = :green, linewidth = 10)

#     # Prawa bramka (x=x_len)
#     lines!(axis, [params.x_len, params.x_len], [goal_ymin, goal_ymax], color = :green, linewidth = 10)

#     # --- OBRAMOWANIE BOISKA --- #
#     # Prostokąt: 4 rogi boiska
#     x_rect = [0.0, params.x_len, params.x_len, 0.0, 0.0]
#     y_rect = [0.0, 0.0, params.y_len, params.y_len, 0.0]
#     lines!(axis, x_rect, y_rect, color = :black, linewidth = 2)

#     scatter_puck = GLMakie.scatter!(axis, xy_puck; color = :red, markersize = 20params.puck_radius)
#     scatter_mallets = GLMakie.scatter!(axis, xy_mallet12; color = :blue, markersize = 20params.mallet_radius)
#     display(scene)

#     for i in eachindex(new_puck_xs)
#         animstep!(axis, xy_puck, xy_mallet12, i, new_puck_xs, new_puck_ys, new_mallet1_xs, new_mallet1_ys, new_mallet2_xs, new_mallet2_ys)
#         sleep(0.005)
#     end

# end
# end