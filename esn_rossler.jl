#todo
# select for replicability from goodstuff
# analyze parameter paterns in ultragoodstuff
# plot turing test scatter with lz, entropy, mse
# make 9x9 samples
# do events code thing

using ReservoirComputing
using OrdinaryDiffEq
using Plots
using Findpeaks
using LaTeXStrings
using JLD

struct SavitzkyGolayFilter{M,N} end
@generated function (::SavitzkyGolayFilter{M,N})(data::AbstractVector{T}) where {M, N, T}
            #Create Jacobian matrix
            J = zeros(2M+1, N+1)
            for i=1:2M+1, j=1:N+1
                J[i, j] = (i-M-1)^(j-1)
            end
            e₁ = zeros(N+1)
            e₁[1] = 1.0

            #Compute filter coefficients
            C = J' \ e₁

            #Evaluate filter on data matrix

            To = typeof(C[1] * one(T)) #Calculate type of output
            expr = quote
                n = size(data, 1)
                smoothed = zeros($To, n)
                @inbounds for i in eachindex(smoothed)
                    smoothed[i] += $(C[M+1])*data[i]
                end
                smoothed
            end

            for j=1:M
                insert!(expr.args[6].args[3].args[2].args, 1,
                    :(if i - $j ≥ 1
                        smoothed[i] += $(C[M+1-j])*data[i-$j]
                      end)
                )
                push!(expr.args[6].args[3].args[2].args,
                    :(if i + $j ≤ n
                        smoothed[i] += $(C[M+1+j])*data[i+$j]
                      end)
                )
            end

            return expr
end

function smooth(sol; window_width = 10, order=3)
    sg = SavitzkyGolayFilter{window_width, order}()
    filtered = vcat(transpose(sg.(sg.(sg.(eachrow(sol)))))...)
end

function blockentropy(str,m)
    blocks=[str[i:i+m-1] for i in 1:length(str)-m-1]
    psm = [count(==(b),blocks) for b in unique(blocks)]./length(blocks)
    return -sum([p*log(p) for p in psm])
end

function marr(str, mx = 10)
    a = map(1:mx+1) do m
        blockentropy(str,m)
    end
    b = [a[1]]
    for i = 2:mx+1
        push!(b, a[i]-a[i-1])
    end
    return b
end

function chaosscore(sol, test)
    sol_str = symbolize(smooth(sol), .1)[1]
    test_str = symbolize(test, .1)[1]
    lz = abs(LZ(sol_str)-LZ(test_str))
    ent = sum([abs2(blockentropy(sol_str,m)-blockentropy(test_str,m)) for m = 2:10])
    symadj = any(<(-1), sol[3,:]) ? 1.0e9 : 0.
    sol_i = sol[:,1:1000]
    test_i = test[:,1:1000]
    mse = reduce(+, (sol_i .- test_i).^2)/length(test_i)
    score = mse * lz * ent/10000 + symadj
    println("lz = $lz, ent = $ent, mse = $mse, score = $score")
    (score, lz, ent, mse)
end

function symbolize(sol, min_prom; min_height = 1.)
    peaks = findpeaks(sol[3,:], min_prom = min_prom, min_height = min_height) |> sort
    valleys = findpeaks(-1 .*sol[3,:], min_prom = min_prom, min_dist = 0 )|> sort
    filter!(i -> sol[1,i]<-sol[2,i], valleys)
    crits = sort(union(peaks,valleys))
    pixs = []
    vixs = []
    symarr = map(crits) do i
        i in peaks ? 1 : 0
    end
    return (join(symarr), [sol[:,i] for i in peaks], [sol[:,i] for i in valleys], peaks, valleys)
end

include("./LempelZiv.jl")
LZ = LempelZiv.lempel_ziv_complexity

##

u0 = [0.2,0.4,0.2]
tspan = (0.0,200000.0)
p= [0.341,0.3,4.8]
#define rossler system
function rossler(du,u,p,t)
    du[1] = -u[2]-u[3]
    du[2] = u[1] + p[1]*u[2]
    du[3] = p[2]+u[3]*(u[1]-p[3])
end

#solve system
prob = ODEProblem(rossler, u0, tspan, p)
sol = solve(prob, BS3(), saveat = .02, maxiters = 50000000)

clean_data = Matrix(hcat(sol.u...))
#symbolize(data, .1) to test length

#add gaussian noise
data = copy(clean_data)
for i in eachindex(clean_data)
    data[i] += .3*randn()
end

## ESN starts here
#model parameters
train_len = 30000
predict_len = 30000
shift = 2000

#partition data into train and test
train = data[:, shift:shift+train_len-1]
test = data[:, shift+train_len:shift+train_len+predict_len-1]

#define search space
pars = (
    approx_res_size = [80],
    radius = .85:.05:1.5,
    activation = [tanh],
    degree = [2,3,4],
    sigma = .05:.01:0.6,
    alpha = .05:.01:1.,
    nla_type = [NLADefault(), NLAT1(), NLAT2(), NLAT3()],
    extended_states = [false, true],
    beta = 0.:.01:0.1,
)

#reduce(*,[length(e) for e in pars]) # size of space

goodstuff = []
for i = 1:10000
    ps = [l[rand(1:end)] for l in pars]
    #create echo state network
    esn = ESN(ps[1],
        train,
        ps[4],
        ps[2],
        activation = ps[3],
        sigma = ps[5],
        alpha = ps[6],
        nla_type = ps[7],
        extended_states = ps[8])
    #training and prediction
    W_out = ESNtrain(esn, ps[9])
    output = ESNpredict(esn, predict_len, W_out)
    scoretup = chaosscore(output, test)
    score = scoretup[1]
    if length(goodstuff) < 50
        if !(isnan(score))
            push!(goodstuff, (score = scoretup, esn = esn, pars = ps, out = output, wout = W_out))
        end
    else
        maxscore = findmax([e.score[1] for e in goodstuff])
        if (score < maxscore[1])
            println("$i replacing $(goodstuff[maxscore[2]].score) with $score")
            goodstuff[maxscore[2]] = (score = scoretup, esn = esn, pars = ps, out = output, wout = W_out)
        end
    end
end

## plots
# no entropies
begin
    gr()
    sfarr = []
    for idx = 1:50
        output = goodstuff[idx].out
        esn = goodstuff[idx].esn
        score, lz, ent, mse = round.(goodstuff[idx].score, sigdigits=2)
        sf = plot(transpose(output)[:,1], transpose(output)[:,2],
            transpose(output)[:,3],
            label="i=$idx score=$score, error=$mse, h=$ent, lz=$lz")
        push!(sfarr, sf)
    end
    plot(sfarr..., layout = (5,10), size = (4000,2000))
end

#with entropies
begin
    gr()
    sfarr = []
    test4 = clean_data[:, shift+train_len:200000]
    for idx = 1:50
        output1 = goodstuff[idx].out
        output4 = ESNpredict(goodstuff[idx].esn, size(test4)[2], goodstuff[idx].wout)
        esn = goodstuff[idx].esn
        score, lz, ent, mse = round.(goodstuff[idx].score, sigdigits=2)
        sf = plot(transpose(output1)[:,1], transpose(output1)[:,2],
            transpose(output1)[:,3],
            label="i=$idx score=$score, error=$mse, h=$ent, lz=$lz",
            ticks = nothing, axis = false)
        actstr = symbolize(test4,.1)
        simstr = symbolize(output4, .1)
        args = (
            inset = (1,bbox(0,0,.8,.8,:center)),
            subplot = 2,
            bg_inside = nothing,
            axis = false,
            ticks = nothing
            )
        scatter!(
            0:10,
            marr(actstr[1], 10);
            label = "actual",
            color = :green,
            markersize = 10,
            markeralpha = .5,
            args...
        )
        scatter!(
            0:10,
            marr(simstr[1], 10),
            markershape = :circle,
            markersize = 10,
            markeralpha = .5,
            markerstrokealpha = 1,
            markerstrokewidth = 3,
            markerstrokecolor = :red,
            label = "predicted",
            subplot = 2,
        )
        push!(sfarr, sf)
    end
    plot(sfarr..., layout = (5,10), size = (5000,2000))

end

#savefig("bigfig.png")


begin
    goodchaos = 13
    badchaos = 32
    longperiodic = 17
    shortperiodic = 35

    idxs = [goodchaos, badchaos, longperiodic, shortperiodic]
    sfarr = []
    test4 = clean_data[:, shift+train_len:200000]
    for (j,idx) in enumerate(idxs)
        output1 = goodstuff[idx].out
        output4 = ESNpredict(goodstuff[idx].esn, size(test4)[2], goodstuff[idx].wout)
        esn = goodstuff[idx].esn
        score, lz, ent, mse = round.(goodstuff[idx].score, sigdigits=2)
        sf = plot(transpose(output1)[:,1], transpose(output1)[:,2],
            transpose(output1)[:,3],
            label="score=$score, error=$mse, h=$ent, lz=$lz",
            ticks = nothing, axis = false, linealpha = .5,
            legend = (.298,.9))
        scatter!([-5],[-5],[-5], color = :white, markersize = 0, label = "")
        actstr = symbolize(test4,.1)
        simstr = symbolize(output4, .1)
        args = (
            inset = (1,bbox(0,0,.75,.75,:center)),
            subplot = 2,
            bg_inside = nothing,
            )
        scatter!(
            0:10,
            marr(actstr[1], 10);
            label = "h actual",
            color = :green,
            markersize = 5,
            markeralpha = .8,
            args...
        )
        scatter!(
            0:10,
            marr(simstr[1], 10),
            markershape = :circle,
            markersize = 5,
            markeralpha = .8,
            markerstrokealpha = 1,
            markerstrokewidth = 1,
            markerstrokecolor = :red,
            label = "h predicted",
            subplot = 2,
            ylabel = "Entropy",
            xlabel = "Block Size",
            xticks = collect(0:10),
            legend = (0.2,0.9)
        )

        actstr = symbolize(test4,.1, min_height = -80.)
        simstr = symbolize(output4, .1,min_height = -80.)
        simmap = [e[3] for e in simstr[2]]
        actmap = [e[3] for e in actstr[2]]

        scatter!(actmap[1:end-1], actmap[2:end], xlabel = L"z\ map", color = :violetred, labelfontsize = 11,
            subplot = 3, bg_inside = nothing, ticks = nothing,
            inset = (1,bbox(0.15,.25,.15*1.8,.2*1.8,:right)), label = "", markersize = 2, labelmargin = 0)
        scatter!(simmap[1:end-1], simmap[2:end],
            label = "",  color = :white, subplot = 3, markersize = 3.3, markerstrokecolor = :black)

        push!(sfarr, sf)
    end
    plot(sfarr..., layout = (2,2), size = (1000,600))

end
savefig("rosslersearchsample.png")

#after handpicking the best idx, try a long simulation to ensure that
#it is not a periodic orbit
idx = 13
esn = goodstuff[idx].esn
output = ESNpredict(esn, 500000, goodstuff[idx].wout)
sf = plot(transpose(output)[:,1], transpose(output)[:,2], transpose(output)[:,3], label=idx)

# create figure with events
test3 = clean_data[:, shift+train_len:100000+shift+train_len-1]#data[:, shift+train_len:end]

#get clean z'=0 points for figs
tspan2= (0.,4000.)
prob2 = ODEProblem(rossler, u0, tspan2, p)
sol2 = solve(prob2, DP8(), dt = .01, adaptive = false)
clean_data2 = Matrix(hcat(sol2.u...))
clean_test = clean_data2[:, 2shift+2train_len:200000+2shift+2train_len-1]

begin
    gr()
    axlbl = (xlabel = L"x", ylabel = L"y", zlabel = L"z")

    out = smooth(output, window_width = 20, order = 1)
    sf = plot(transpose(out)[:,1], transpose(out)[:,2], transpose(out)[:,3],
        color = :black, label=""; axlbl...)
    s = symbolize(out, .1)
    abovepoints = s[2] |> Iterators.flatten |> collect |> x -> reshape(x,(3,length(s[2])))
    belowpoints = s[3] |> Iterators.flatten |> collect |> x -> reshape(x,(3,length(s[3])))
    scatter3d!(abovepoints[1,:], abovepoints[2,:], abovepoints[3,:],
     camera = (30,30), label = "", color = :orange)
    scatter3d!(belowpoints[1,:], belowpoints[2,:], belowpoints[3,:],
     camera = (30,30), label = "", color = :steelblue)
    pt = plot(transpose(out)[1:10000,3], label = "", color = :black)
    pxs = (filter(x -> x < 10000, s[4]))
    vxs = (filter(x -> x < 10000, s[5]))
    scatter!(pxs,[out[3,i] for i in pxs], label = "", color = :orange)
    scatter!(vxs,[out[3,i] for i in vxs], label = "", xlabel = L"t", ylabel = L"z", color = :steelblue)



    sf2 = plot(transpose(test3)[:,1], transpose(test3)[:,2], transpose(test3)[:,3], label="actual",
        color = :violetred ; axlbl...)
    plot3d!([.0,.0001], [.0,.0001], [.0,.0001], label = "predicted", color = :black)
    s2 = symbolize(clean_test, .1)

    abovepoints = s2[2] |> Iterators.flatten |> collect |> x -> reshape(x,(3,length(s2[2])))
    belowpoints = s2[3] |> Iterators.flatten |> collect |> x -> reshape(x,(3,length(s2[3])))
    scatter3d!(abovepoints[1,:], abovepoints[2,:], abovepoints[3,:],
     camera = (30,30), label = "above", color = :orange)
    scatter3d!(belowpoints[1,:], belowpoints[2,:], belowpoints[3,:],
     camera = (30,30), label = "below", color = :steelblue)

    s3 = symbolize(test3, .1)

    pt2 = plot(transpose(test3)[1:10000,3], label = "", color = :violetred)
    pxs = (filter(x -> x < 10000, s3[4]))
    vxs = (filter(x -> x < 10000, s3[5]))
    scatter!(pxs,[test3[3,i] for i in pxs], label = "", color = :orange)
    scatter!(vxs,[test3[3,i] for i in vxs], label = "", xlabel = L"t", ylabel = L"z", color = :steelblue)

    # generate zmaps
    ss = symbolize(output, .125, min_height = -80.)
    ss2 = symbolize(test3, .051, min_height = -1.)


    simmap = [e[3] for e in ss[2]]
    actmap = [e[3] for e in ss2[2]]

    scatter(actmap[1:end-1], actmap[2:end],
        label = "actual", xlabel = L"z\ max_{n}", ylabel = L"z\ max_{n+1}", color = :violetred)
    scatter!(simmap[1:end-1], simmap[2:end],
        label = "",  color = :black, markeralpha = .5)
    pm = scatter!([simmap[1]], [simmap[2]], label = "predicted", color = :black, )


    l = @layout  [a b c ; grid(1,2){.2h}]
    final = plot(sf2,pm,sf,pt2,pt, layout= l, size =(1200,500),
        margin = 5Plots.mm, markerstrokewidth = 0.4,bottommargin = 5Plots.mm,
        topmargin = 0Plots.mm, title = ["A" "B" "C" "D" "E"], titlelocation = :left)
end

savefig(final, "rosslerfig.png")

begin
    test5 = clean_data[:, shift+train_len:end]
    output5 = ESNpredict(goodstuff[idx].esn, size(test5)[2], goodstuff[idx].wout)
    actstr = symbolize(test5,.1)
    simstr = symbolize(output5, .1)
    gr()
    entfig = scatter(
        0:10,
        marr(actstr[1], 10),
        label = "actual",
    )
    scatter!(
        0:10,
        marr(simstr[1], 10),
        markershape = :circle,
        markersize = 10,
        markeralpha = .3,
        markerstrokealpha = 1,
        markerstrokewidth = 3,
        markerstrokecolor = :red,
        label = "predicted",
        ylabel = "entropy",
        xlabel = "block size")
end
savefig("entropyfig.png")

#savedata
save("goodstuff.jld", "goodstuff", goodstuff[idx].esn)
goodstuff[idx].pars

a =[e.pars for e in goodstuff]


##
# generate long output string

open("rossler_actual_str.txt", "w") do io
   println(io,join([e*" " for e in actstr[1]])
)
end

open("rossler_predicted_str.txt", "w") do io
   println(io,join([e*" " for e in simstr[1]])
)
end

#save("rossler_julia_variables.jld","esn",goodstuff[idx].esn)
#load("rossler_julia_variables.jld")

using DelimitedFiles

r30fake = Int.(readdlm("/home/jscully2@gsuad.gsu.edu/Dropbox/Lorenz_&_Rossler/Lz/r30_fake.dat"))
r30real = Int.(readdlm("/home/jscully2@gsuad.gsu.edu/Dropbox/Lorenz_&_Rossler/Lz/r30_true.dat"))
entropyplot(r30fake,r30real, title = "r=30")

r75fake = Int.(readdlm("/home/jscully2@gsuad.gsu.edu/Dropbox/Lorenz_&_Rossler/Lz/r75_fake.dat"))
r75real = Int.(readdlm("/home/jscully2@gsuad.gsu.edu/Dropbox/Lorenz_&_Rossler/Lz/r75_true.dat"))
entropyplot(r75fake,r75real, title = "r=75")

r925fake = Int.(readdlm("/home/jscully2@gsuad.gsu.edu/Dropbox/Lorenz_&_Rossler/Lz/r925_fake.dat"))
r925real = Int.(readdlm("/home/jscully2@gsuad.gsu.edu/Dropbox/Lorenz_&_Rossler/Lz/r925_true.dat"))
entropyplot(r925fake,r925real)

load("goodstuff.jld")
