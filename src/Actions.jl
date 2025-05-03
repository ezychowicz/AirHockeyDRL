module Actions
Base.@kwdef struct Action #to sa wspolrzedne biegunowe wektora dv
    dv_angle::Float32 #kat: zakres [-pi, pi] 
    dv_len::Float32 #dlugosc wektora
end

function map_output_to_action(output::Vector{Float32}, env::AirHockeyEnv)::Action
    """
    map from normalized form to real form: 
    - v_dir: [-1,1] -> [-pi,pi]
    - v_len: [-1,1] -> [0, MAX_DV_LEN] 
    """ 

    action = Action(
        dv_angle = ,
        dv_len = 
    )
    return action
end