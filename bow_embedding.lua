--BOW 
require 'nn'
--require 'cunn'
-- IMP if args is not passed, it takes from global 'args'
return function(args)

    function create_network(args)
        local n_hid = 100
        local mlp = nn.Sequential()
        mlp:add(nn.Reshape(args.hist_len*args.ncols*args.state_dim))
        mlp:add(nn.Linear(args.hist_len*args.ncols*args.state_dim, n_hid))
        mlp:add(nn.Rectifier())
        mlp:add(nn.Linear(n_hid, n_hid))
        mlp:add(nn.Rectifier())

        action_layer = nn.Linear(n_hid, args.n_actions)

        local mlp_out = nn.ConcatTable()        
        mlp_out:add(nn.Identity())
        local action_object_nn = nn.Sequential()
        action_object_nn:add(nn.Linear(args.n_actions, n_hid))
        action_object_nn:add(nn.Linear(n_hid, args.n_objects))
        mlp_out:add(action_object_nn)

        mlp:add(action_layer)
        mlp:add(mlp_out)
        
        if args.gpu >=0 then
            mlp:cuda()
        end    
        return mlp
    end

    return create_network(args)
end
