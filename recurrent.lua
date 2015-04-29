
function lstm(i, prev_c, prev_h, layer_size_in, layer_size_out)
  function new_input_sum()    
    local i2h            = nn.Linear(layer_size_in, layer_size_out)
    local h2h            = nn.Linear(layer_size_out,layer_size_out)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end

  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end


function create_encoder(input,prev_s)
  -- i = {[0] = input}
  --global embedding
  embedding = Embedding(#dataset_metainfo["symbols"],params.input_size)

  local i                = {[0] = embedding(input)}

  local next_s           = {}
  local splitted         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = splitted[2 * layer_idx - 1]
    local prev_h         = splitted[2 * layer_idx]
    local dropped = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h, params.input_size, params.rnn_size)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end

  local y  = nn.Linear(params.rnn_size, params.encout_size)(i[params.layers])
  return next_s, y
end


function create_rnn()
  local input = nn.Identity()()
  local target = nn.Identity()()


  mlp:add(nn.Reshape(args.hist_len*args.ncols*args.state_dim))


  ------------ encoder ------------
  local enc_prev_s = nn.Identity()()
  local enc_next_s, enc_y
  enc_next_s, enc_y = create_encoder(input, enc_prev_s)

  local module           = nn.gModule({input, target, enc_prev_s, dec_prev_s}, 
                                      {pred, KLDerr, nn.Identity()(enc_next_s), nn.Identity()(dec_next_s), enc_y})

  -- print(input, target, enc_prev_s, dec_prev_s, Z_mean, Z_cov, err, nn.Identity()(enc_next_s), nn.Identity()(dec_next_s))
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  if CUDA_ON then
    module:cuda()
  end
  return module
end