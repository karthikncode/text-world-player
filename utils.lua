local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

function split(s, pattern)
	local parts = {}
	for i in string.gmatch(s, pattern) do
  	table.insert(parts, i)
	end
	return parts
end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function string.trim(s)
  -- return (s:gsub("^%s*(.-)%s*$", "%1"))
  return s:match "^%W*(.-)%s*$"
end

function reverse_tensor(tensor)
  --make sure tensor is 1D
  local n = tensor:size(1)
  local tmp = torch.Tensor(n)
  for i=1, n do
    tmp[i] = tensor[n+1-i]
  end
  return tmp
end

-- function specific to make available_objects tensor
function table_to_binary_tensor(t,N)   
  local tensor
  if t then
    tensor = torch.zeros(N)
    for i,val in pairs(t) do
      tensor[val] = 1
    end
  else
    tensor = torch.ones(N)
  end
  return tensor
end



function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

-- IMP: very specific function - do not use for arbitrary tensors
function tensor_to_table(tensor, state_dim, hist_len)
  batch_size = tensor:size(1)
  local NULL_INDEX = #symbols+1 

  -- convert 0 to NULL_INDEX (this happens when hist doesn't go back as far as hist_len in chain)
  for i=1, tensor:size(1) do
    for j=1, tensor:size(2) do
      if tensor[i][j] == 0 then
        tensor[i][j] = NULL_INDEX
      end
    end
  end


  local t2 = {}

  if tensor:size(1) == hist_len then
    -- hacky: this is testing case. They don't seem to have a consistent representation
    -- so this will have to do for now.
    -- print('testing' , tensor:size())
    for j=1, tensor:size(1) do
      for k=1, tensor:size(2)/state_dim do
        t2_tmp = {}
        for i=(k-1)*state_dim+1, k*state_dim do
          t2_tmp[i%state_dim] = tensor[{{j}, {i}}]:reshape(1)
        end
        t2_tmp[state_dim] = t2_tmp[0]
        t2_tmp[0] = nil
        table.insert(t2, t2_tmp)
      end
    end
  else
    -- print('training' , tensor:size())
    -- print(tensor[{{1}, {}}])
    for j=1, tensor:size(2)/state_dim do
      t2_tmp = {}
      for i=(j-1)*state_dim+1,j*state_dim do
        t2_tmp[i%state_dim] = tensor[{{}, {i}}]:reshape(batch_size)   
      end
      t2_tmp[state_dim] = t2_tmp[0]
      t2_tmp[0] = nil
      table.insert(t2, t2_tmp)   
    end
  end

  -- for i=1, #t2 do
  --   for j=1, #t2[1] do
  --     for k=1, t2[i][j]:size(1) do
  --       assert(t2[i][j][k] ~= 0, "0 element at"..i..' '..j..' '..k)
  --     end
  --   end
  -- end

  return t2
end


function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

function table.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end
