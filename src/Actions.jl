
function action(p::RandomPolicy, state::State)
    Action(
        rand(p.distribution_angle),
        rand(p.distribution_len))
end